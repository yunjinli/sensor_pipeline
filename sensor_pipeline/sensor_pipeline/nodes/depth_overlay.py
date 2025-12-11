#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
from message_filters import Subscriber, ApproximateTimeSynchronizer

from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import cv2
import sys
from pathlib import Path
import os

class DepthOverlay(Node):
    def __init__(self):
        super().__init__('depth_overlay')

        # Params
        self.declare_parameter('pc_topic', '/pmd_royale_ros_camera_node/point_cloud_0')
        self.declare_parameter('rgb_image_topic', '/rgb/image_rect_color')
        self.declare_parameter('rgb_camera_info_topic', '/camera/realsense/color/camera_info_over')
        self.declare_parameter('rgb_optical_frame', 'camera_color_optical_frame')
        self.declare_parameter('queue_size', 3)

        pc_topic   = self.get_parameter('pc_topic').get_parameter_value().string_value
        img_topic  = self.get_parameter('rgb_image_topic').get_parameter_value().string_value
        info_topic = self.get_parameter('rgb_camera_info_topic').get_parameter_value().string_value
        self.rgb_frame = self.get_parameter('rgb_optical_frame').get_parameter_value().string_value
        q = int(self.get_parameter('queue_size').get_parameter_value().integer_value)

        # self.pub_dbg = self.create_publisher(Image, '/rgb/debug_projection', 1)
        
        # TF buffer/listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.time.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Bridge
        self.bridge = CvBridge()

        # QoS: images often BestEffort; PC usually Reliable
        img_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=2,
                             reliability=QoSReliabilityPolicy.BEST_EFFORT)
        pc_qos  = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=2,
                             reliability=QoSReliabilityPolicy.BEST_EFFORT)

        # Subscribers (message_filters for approx sync on image+pc; CameraInfo latched via plain sub)
        self.sub_pc  = Subscriber(self, PointCloud2, pc_topic, qos_profile=pc_qos)
        self.sub_img = Subscriber(self, Image, img_topic, qos_profile=img_qos)

        self.K = None
        self.width = None
        self.height = None
        self.sub_info = self.create_subscription(CameraInfo, info_topic, self.on_info, 10)

        self.sync = ApproximateTimeSynchronizer([self.sub_pc, self.sub_img], queue_size=q, slop=0.03)
        self.sync.registerCallback(self.on_sync)

        # Publisher (Reliable)
        out_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                             reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.VOLATILE)
        # self.pub = self.create_publisher(PointCloud2, out_topic, out_qos)
        out_topic = img_topic.replace('image_rect_color', 'image_rect_depth')
        self.pub_depth_map = self.create_publisher(Image, out_topic, out_qos)
        
        self.get_logger().info(f'Depth overlay with RGB: pc={pc_topic}, img={img_topic}, info={info_topic} → {out_topic}')

        self.save = False
        self.R = None
        self.T = None
        self.tf = None
    
    def imgmsg_to_rgb(self, img_msg: Image) -> np.ndarray:
        """
        Convert sensor_msgs/Image to RGB uint8 HxWx3 NumPy array.
        Handles common encodings: rgb8, bgr8, rgba8, bgra8.
        """
        enc = img_msg.encoding.lower()
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        if enc.startswith("bgr"):
            # convert to RGB for consistent saving
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        else:
            # already RGB if desired_encoding="rgb8"
            rgb = cv_img
        return rgb

    def publish_depth_buffer(self, rgb_msg: Image, depth_numpy):
        try:
            if depth_numpy.dtype != np.float32:
                depth_numpy = depth_numpy.astype(np.float32)

            ros_msg = self.bridge.cv2_to_imgmsg(depth_numpy, encoding="32FC1")

            stamp = rgb_msg.header.stamp
            frame_id = rgb_msg.header.frame_id
            
            ros_msg.header.stamp = stamp
            ros_msg.header.frame_id = frame_id

            self.pub_depth_map.publish(ros_msg)

        except Exception as e:
            self.get_logger().error(f'Failed to publish depth: {str(e)}')
            
    def on_info(self, ci: CameraInfo):
        # Expect rectified model (D=0). We use K (or P[0:3,0:3]) as intrinsics
        K = np.array(ci.k, dtype=np.float64).reshape(3,3)
        P = np.array(ci.p, dtype=np.float64).reshape(3,4)
        # Prefer P’s left 3x3 if nonzero; else K
        if not np.allclose(P[:3,:3], 0):
            self.K = P[:3,:3].copy()
        else:
            self.K = K.copy()
        self.width  = int(ci.width)
        self.height = int(ci.height)
    
    def pc2_xyz_numpy(self, msg: PointCloud2):
        # Assumes little-endian floats (most ROS cams); add checks if needed
        assert not msg.is_bigendian, "Only little-endian PointCloud2 supported"
        # Find field offsets
        off = {f.name: f.offset for f in msg.fields}
        for k in ('x','y','z'):
            if k not in off: raise ValueError("PointCloud2 missing XYZ fields")

        # Reshape raw bytes to (N, point_step) uint8 view
        N = msg.width * msg.height if msg.height > 1 else msg.width
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        if buf.size < N * msg.point_step:
            N = buf.size // msg.point_step
        view = buf.reshape(N, msg.point_step)

        # Zero-copy float32 views for each channel
        # x = view[:, off['x']:off['x']+4].view('<f4').ravel()
        # y = view[:, off['y']:off['y']+4].view('<f4').ravel()
        # z = view[:, off['z']:off['z']+4].view('<f4').ravel()
        x = view[:, off['x']:off['x']+4].copy().view('<f4').ravel()
        y = view[:, off['y']:off['y']+4].copy().view('<f4').ravel()
        z = view[:, off['z']:off['z']+4].copy().view('<f4').ravel()
        # Stack without copying (one contiguous copy at the end)
        xyz = np.empty((N,3), dtype=np.float32)
        xyz[:,0] = x; xyz[:,1] = y; xyz[:,2] = z
        # Optionally drop NaNs here:
        # m = np.isfinite(xyz).all(1)
        # return xyz[m]
        return xyz
    
    def transform_pcd(self, pc_msg: PointCloud2, src):
        if self.tf is None:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.rgb_frame,               # target
                    src,                          # source
                    rclpy.time.Time(),            # (static TF -> time 0 is fine)
                    timeout=rclpy.duration.Duration(seconds=0.2)
                )
            except Exception as e:
                self.get_logger().warn(f'TF lookup failed: {e}')
                return
            self.tf = tf
        
        pc_rgb = do_transform_cloud(pc_msg, self.tf)   # ← transform the whole cloud
        
        return pc_rgb
    
    def cache_R_T_from_tf(self, src):
        if self.R is None or self.T is None:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.rgb_frame,               # target
                    src,                          # source
                    rclpy.time.Time(),            # (static TF -> time 0 is fine)
                    timeout=rclpy.duration.Duration(seconds=0.2)
                )
            except Exception as e:
                self.get_logger().warn(f'TF lookup failed: {e}')
                return
            t = tf.transform.translation
            q = tf.transform.rotation
            # quaternion to rotation (OpenCV/OpenGL-style)
            x, y, z, w = q.x, q.y, q.z, q.w
            self.R = np.array([
                [1-2*(y*y+z*z),   2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
                [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
            ], dtype=np.float32)
            self.T = np.array([t.x, t.y, t.z], dtype=np.float32)
        else:
            return
        
    def on_sync(self, pc_msg: PointCloud2, img_msg: Image):
        if self.K is None:
            self.get_logger().warn('No CameraInfo received yet; skipping frame')
            return

        # Convert image
        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge failed: {e}')
            return
        if img.shape[1] != self.width or img.shape[0] != self.height:
            self.get_logger().warn('Image size != CameraInfo size; skipping')
            return

        # if not self.save:
        #     self.get_logger().info('Saving one frame for debugging...')
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     cv2.imwrite('/home/yunjinli/camera_calibration/dbg_image.png', img)
        #     pc_points = np.asarray([[p[0], p[1], p[2]]
        #                     for p in pc2.read_points(pc_msg, field_names=('x','y','z'), skip_nans=True)],
        #                     dtype=np.float32)
        #     np.save('/home/yunjinli/camera_calibration/dbg_pc_xyz.npy', pc_points)
        #     self.save = True
        src = pc_msg.header.frame_id
        
        # if src != self.rgb_frame:
        #     self.cache_R_T_from_tf(src=src)
        #     Ptof = self.pc2_xyz_numpy(pc_msg)
        #     Prgb = (self.R @ Ptof.T + self.T.reshape(3, 1)).T
        # else:
        #     # self.get_logger().info('PC already in RGB frame')
        #     pc_rgb = pc_msg  # already in RGB frame
        #     Prgb = self.pc2_xyz_numpy(pc_rgb)
        
        if src != self.rgb_frame:
            pc_rgb = self.transform_pcd(pc_msg=pc_msg, src=src)
            Prgb = self.pc2_xyz_numpy(pc_rgb)
        else:
            # self.get_logger().info('PC already in RGB frame')
            pc_rgb = pc_msg  # already in RGB frame
            Prgb = self.pc2_xyz_numpy(pc_rgb)
        
        # Now read points from pc_rgb; they’re guaranteed in RGB optical frame:
        
        if Prgb.size == 0: return
        Prgb = Prgb.astype(np.float32, copy=False)
        X, Y, Z = Prgb[:, 0], Prgb[:, 1], Prgb[:, 2]      
        
        fx, fy = self.K[0,0], self.K[1,1]
        cx, cy = self.K[0,2], self.K[1,2]

        uf = (fx * (X / Z) + cx)
        vf = (fy * (Y / Z) + cy)
        
        valid = (uf >= 0) & (uf < self.width) & (vf >= 0) & (vf < self.height)
        
        # self.get_logger().info(f'Projected points: {len(uf)} total, {valid.sum()} in image bounds')
        # if not np.any(valid):
        # idx = np.nonzero(in_bounds)
        if valid.sum() == 0:
            self.get_logger().warn('No projected points inside image; skipping')
            return
        
        u = uf.astype(np.int32)
        v = vf.astype(np.int32)
        dbg = img.copy()
        uu = u[valid]; vv = v[valid]
        # 1) robust depth range (avoid a few crazy far points compressing colors)
        z_valid = Z[valid]
        z = z_valid.astype(np.float32)
        zmin, zmax = np.percentile(z, [2, 98])  # tweak if needed
        if zmax <= zmin:  # fallback
            zmin, zmax = float(z.min()), float(z.max())

        # 2) normalize to [0, 255] (optionally invert if you want near=red)
        zn = np.clip((z - zmin) / (zmax - zmin + 1e-6), 0, 1)
        zn8 = (zn * 255).astype(np.uint8).reshape(-1, 1)  # shape (N,1)

        # 3) apply a colormap (JET is widely available; TURBO if your OpenCV has it)
        colors = cv2.applyColorMap(zn8, cv2.COLORMAP_JET)  # BGR uint8, shape (N,3)
        colors = np.squeeze(colors, axis=1)
        # 4) optionally downsample for clarity
        mask_vis = np.zeros(len(uu), dtype=bool)
        mask_vis[::1] = True  # every 10th point; adjust density

        # ----- Dense depth visualization (replace your marker loop) -----
        H, W = img.shape[:2]

        # 1) per-pixel nearest depth (vectorized)
        uu_i = np.clip(np.round(uf[valid]).astype(np.int32), 0, W-1)
        vv_i = np.clip(np.round(vf[valid]).astype(np.int32), 0, H-1)
        z     = Z[valid].astype(np.float32)

        Zimg = np.full((H, W), np.float32(np.inf), dtype=np.float32)
        flat = (vv_i.astype(np.int64) * W + uu_i.astype(np.int64))
        Zflat = Zimg.ravel()
        # take min depth for pixels hit by multiple points
        np.minimum.at(Zflat, flat, z)
        Zimg = Zflat.reshape(H, W)

        # 2) (optional) fast hole fill via box-filter average of finite neighbors
        finite = np.isfinite(Zimg)
        if not np.all(finite):
            Z0   = Zimg.copy(); Z0[~finite] = 0.0
            mask = finite.astype(np.float32)

            # 5x5 average of neighbors (fast); tune kernel if you like
            ksz = 5
            num = cv2.blur(Z0,   (ksz, ksz))
            den = cv2.blur(mask, (ksz, ksz))
            Zfill = np.where(den > 1e-6, num / np.maximum(den, 1e-6), Zimg)
        else:
            Zfill = Zimg

        self.publish_depth_buffer(rgb_msg=img_msg, depth_numpy=Zfill)
        
def main():
    rclpy.init()
    rclpy.spin(DepthOverlay())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
