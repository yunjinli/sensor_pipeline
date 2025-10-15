#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from cv_bridge import CvBridge
import numpy as np
from message_filters import Subscriber, ApproximateTimeSynchronizer

class ColorizeToFGray(Node):
    def __init__(self):
        super().__init__('colorize_tof_gray')

        self.declare_parameter('pc_topic',   '/pmd_royale_ros_camera_node/point_cloud_0')
        # self.declare_parameter('img_topic',  '/pmd_royale_ros_camera_node/gray_image_0')
        self.declare_parameter('img_topic',  '/tof/gray_image_rect')
        # self.declare_parameter('info_topic', '/pmd_royale_ros_camera_node/camera_info')
        self.declare_parameter('info_topic', '/pmd_royale_ros_camera_node/camera_info_over')
        self.declare_parameter('out_topic',  '/tof/points_gray_rgb')
        self.declare_parameter('queue_size', 20)

        pc_topic   = self.get_parameter('pc_topic').get_parameter_value().string_value
        img_topic  = self.get_parameter('img_topic').get_parameter_value().string_value
        info_topic = self.get_parameter('info_topic').get_parameter_value().string_value
        out_topic  = self.get_parameter('out_topic').get_parameter_value().string_value
        q          = int(self.get_parameter('queue_size').get_parameter_value().integer_value)

        # QoS: driver likely uses Reliable for cloud, BestEffort for image
        qos_pc  = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        qos_img = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                             reliability=QoSReliabilityPolicy.BEST_EFFORT)

        self.sub_pc  = Subscriber(self, PointCloud2, pc_topic,  qos_profile=qos_pc)
        self.sub_img = Subscriber(self, Image,       img_topic, qos_profile=qos_img)

        self.bridge = CvBridge()
        self.K = None
        self.W = None
        self.H = None
        self.create_subscription(CameraInfo, info_topic, self.on_info, 10)

        self.sync = ApproximateTimeSynchronizer([self.sub_pc, self.sub_img], queue_size=q, slop=0.05)
        self.sync.registerCallback(self.on_sync)

        out_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                             reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.VOLATILE)
        self.pub = self.create_publisher(PointCloud2, out_topic, out_qos)

        self.get_logger().info(f'Colorizing ToF cloud with ToF gray: {pc_topic} + {img_topic} → {out_topic}')

    def on_info(self, ci: CameraInfo):
        # Use rectified intrinsics if provided in P; else K
        P = np.array(ci.p, dtype=np.float64).reshape(3,4)
        if not np.allclose(P[:3,:3], 0):
            self.K = P[:3,:3].copy()
        else:
            self.K = np.array(ci.k, dtype=np.float64).reshape(3,3)
        self.W, self.H = int(ci.width), int(ci.height)

    def on_sync(self, pc_msg: PointCloud2, img_msg: Image):
        if self.K is None:
            return
        # Convert MONO8 image
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='mono8')
        if img.shape[1] != self.W or img.shape[0] != self.H:
            self.get_logger().warn('Image size != CameraInfo size; skipping')
            return

        # Read points (x,y,z). NOTE: driver publishes fields x,y,z,conf
        pts = np.asarray(
            [ [p[0], p[1], p[2]] for p in pc2.read_points(pc_msg, field_names=('x','y','z'), skip_nans=True) ],
            dtype=np.float32
        )
        if pts.size == 0:
            return

        # Project to pixels (points are already in the ToF optical frame)
        X = pts[:,0].astype(np.float64)
        Y = pts[:,1].astype(np.float64)
        Z = pts[:,2].astype(np.float64)
        valid = Z > 0
        if not np.any(valid):
            return

        fx, fy = self.K[0,0], self.K[1,1]
        cx, cy = self.K[0,2], self.K[1,2]
        u = (fx * (X[valid] / Z[valid]) + cx).astype(np.int32)
        v = (fy * (Y[valid] / Z[valid]) + cy).astype(np.int32)
        inb = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)

        idx = np.nonzero(valid)[0][inb]
        if idx.size == 0:
            return

        u_sel = u[inb]
        v_sel = v[inb]
        gray = img[v_sel, u_sel].astype(np.uint8)

        # Build colored cloud: pack gray as RGB (r=g=b=gray)
        pts_sel = pts[idx,:].astype(np.float32)
        rgb_u32 = (gray.astype(np.uint32) << 16) | (gray.astype(np.uint32) << 8) | gray.astype(np.uint32)
        rgb_f32 = rgb_u32.view(np.float32)

        fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        header = pc_msg.header  # keep ToF optical frame
        cloud_out = pc2.create_cloud(header, fields, np.c_[pts_sel, rgb_f32].tolist())
        self.pub.publish(cloud_out)

def main():
    rclpy.init()
    rclpy.spin(ColorizeToFGray())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
