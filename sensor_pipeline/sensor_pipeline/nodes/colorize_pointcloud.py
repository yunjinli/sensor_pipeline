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


class PointCloudColorizer(Node):
    def __init__(self):
        super().__init__('pointcloud_colorizer')

        # Params
        self.declare_parameter('pc_topic', '/pmd_royale_ros_camera_node/point_cloud_0')
        self.declare_parameter('rgb_image_topic', '/rgb/image_rect_color')
        self.declare_parameter('rgb_camera_info_topic', '/camera/realsense/color/camera_info_over')
        self.declare_parameter('rgb_optical_frame', 'camera_color_optical_frame')
        self.declare_parameter('output_topic', '/rgb/points_xyzrgb_direct')
        self.declare_parameter('queue_size', 3)

        pc_topic   = self.get_parameter('pc_topic').get_parameter_value().string_value
        img_topic  = self.get_parameter('rgb_image_topic').get_parameter_value().string_value
        info_topic = self.get_parameter('rgb_camera_info_topic').get_parameter_value().string_value
        self.rgb_frame = self.get_parameter('rgb_optical_frame').get_parameter_value().string_value
        out_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        q = int(self.get_parameter('queue_size').get_parameter_value().integer_value)

        self.pub_dbg = self.create_publisher(Image, '/rgb/debug_projection', 1)
        
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
        self.pub = self.create_publisher(PointCloud2, out_topic, out_qos)

        self.get_logger().info(f'Colorizing PC: pc={pc_topic}, img={img_topic}, info={info_topic} → {out_topic}')

        self.save = False
        self.R = None
        self.T = None
        self.tf = None
        
        
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

    def bilinear_sample_bgr(self, img, u_f, v_f):
        """
        img: HxWx3 uint8 (BGR)
        u_f, v_f: float pixel coords, shape (K,)
        returns: (K,3) uint8 BGR colors
        """
        H, W, _ = img.shape
        # ensure in-bounds for sampling the 4 neighbors
        u = np.clip(u_f, 0, W - 1.001).astype(np.float32)
        v = np.clip(v_f, 0, H - 1.001).astype(np.float32)

        u0 = np.floor(u).astype(np.int32)
        v0 = np.floor(v).astype(np.int32)
        u1 = np.clip(u0 + 1, 0, W - 1)
        v1 = np.clip(v0 + 1, 0, H - 1)

        du = (u - u0).astype(np.float32)             # (K,)
        dv = (v - v0).astype(np.float32)             # (K,)

        I00 = img[v0, u0].astype(np.float32)         # (K,3)
        I10 = img[v0, u1].astype(np.float32)
        I01 = img[v1, u0].astype(np.float32)
        I11 = img[v1, u1].astype(np.float32)

        w00 = ((1 - du) * (1 - dv))[:, None]         # (K,1)
        w10 = (du * (1 - dv))[:, None]
        w01 = ((1 - du) * dv)[:, None]
        w11 = (du * dv)[:, None]

        out = I00 * w00 + I10 * w10 + I01 * w01 + I11 * w11
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)
    
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
        x = view[:, off['x']:off['x']+4].view('<f4').ravel()
        y = view[:, off['y']:off['y']+4].view('<f4').ravel()
        z = view[:, off['z']:off['z']+4].view('<f4').ravel()

        # Stack without copying (one contiguous copy at the end)
        xyz = np.empty((N,3), dtype=np.float32)
        xyz[:,0] = x; xyz[:,1] = y; xyz[:,2] = z
        # Optionally drop NaNs here:
        # m = np.isfinite(xyz).all(1)
        # return xyz[m]
        return xyz
    
    def publish_xyzrgb_fast(self, header, xyz_f32: np.ndarray, rgb_float: np.ndarray):
        # xyz_f32: (N,3) float32 ; rgb_float: (N,) float32 (packed RGB)
        N = xyz_f32.shape[0]
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width  = N
        msg.is_bigendian = False
        msg.is_dense = True
        msg.fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 16
        msg.row_step   = msg.point_step * N

        # Create a (N,4) float32 array and dump as bytes
        buf = np.empty((N,4), dtype='<f4')
        buf[:, :3] = xyz_f32
        buf[:, 3]  = rgb_float.astype('<f4', copy=False)
        msg.data = buf.tobytes()  # one memcpy

        self.pub.publish(msg)
    
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

        if not self.save:
            self.get_logger().info('Saving one frame for debugging...')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite('/home/yunjinli/camera_calibration/dbg_image.png', img)
            pc_points = np.asarray([[p[0], p[1], p[2]]
                            for p in pc2.read_points(pc_msg, field_names=('x','y','z'), skip_nans=True)],
                            dtype=np.float32)
            np.save('/home/yunjinli/camera_calibration/dbg_pc_xyz.npy', pc_points)
            self.save = True
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

        # 3) normalize to a stable display range and colorize
        NEAR, FAR = 0.1, 10.0  # meters (set once for your setup)
        zn = np.clip((Zfill - NEAR) / (FAR - NEAR), 0.0, 1.0)
        # choose whether near=bright or near=dark by inverting
        depth_u8 = np.uint8((1.0 - zn) * 255)          # near = warm/bright
        cmap = getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET)
        depth_color = cv2.applyColorMap(depth_u8, cmap)  # HxWx3 BGR

        # 4) overlay on RGB (or publish the depth_color alone)
        ALPHA = 0.55
        dbg = cv2.addWeighted(img, 1.0 - ALPHA, depth_color, ALPHA, 0.0)

        # (optional) add a tiny colorbar
        h, w = 200, 12
        bar = np.linspace(255, 0, h, dtype=np.uint8).reshape(h,1)
        bar = cv2.applyColorMap(bar, cmap)
        x0 = dbg.shape[1]-w-10; y0 = 10
        dbg[y0:y0+h, x0:x0+w] = cv2.resize(bar, (w,h), interpolation=cv2.INTER_NEAREST)
        cv2.putText(dbg, f"{NEAR:.2f} m", (x0-60, y0+h),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(dbg, f"{FAR:.2f} m",  (x0-60, y0+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

        self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8'))
        
        colors_bgr = self.bilinear_sample_bgr(img, uf[valid], vf[valid])
        colors_rgb = colors_bgr[:, ::-1].astype(np.uint8)  # BGR->RGB

        # Build colored cloud from the kept 3D points
        # P_sel = Prgb[sel_idx, :].astype(np.float32)
        P_sel = Prgb[valid, :].astype(np.float32)
        # self.get_logger().info(f'Output points: {P_sel.shape[0]}')
        rgb_packed = (colors_rgb[:,0].astype(np.uint32) << 16) | \
                     (colors_rgb[:,1].astype(np.uint32) << 8)  | \
                      colors_rgb[:,2].astype(np.uint32)
        rgb_as_float = rgb_packed.view(np.float32)

        # Create PointCloud2 (fields: x,y,z,rgb)
        self.publish_xyzrgb_fast(header=pc_msg.header, xyz_f32=P_sel, rgb_float=rgb_as_float)
        
def main():
    rclpy.init()
    rclpy.spin(PointCloudColorizer())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
