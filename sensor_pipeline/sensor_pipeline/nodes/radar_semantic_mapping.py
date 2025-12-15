#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import json
import math
import tf2_ros
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
from sensor_msgs_py import point_cloud2 as pc2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

# --- Configuration ---
COLOR_MAP = {
    'grass':    [0, 255, 0],    # Green
    'asphalt':  [50, 50, 50],   # Dark Gray
    'dirt':     [139, 69, 19],  # Brown
    'pavement': [200, 200, 200] # Light Gray
}

class RadarSemanticSurfaceMapping(Node):
    def __init__(self):
        super().__init__('radar_semantic_surface_mapping')

        # Parameters
        self.declare_parameter('radar_frame', 'radar_link')
        self.declare_parameter('fov_deg', 150.0)
        self.declare_parameter('max_range', 1.0)
        
        self.radar_frame = self.get_parameter('radar_frame').value
        self.fov_rad = math.radians(self.get_parameter('fov_deg').value)
        self.max_range = self.get_parameter('max_range').value

        # State
        self.global_points = None   # Nx3 numpy array (XYZ)
        self.global_colors = None   # Nx3 numpy array (RGB)
        self.kdtree = None          # Scipy cKDTree for fast search
        self.map_header = None
        
        self.latest_prediction = None
        self.needs_republish = False

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # QoS
        qos_latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        # Subscribers
        self.create_subscription(PointCloud2, '/rtabmap/cloud_map', self.map_cb, qos_latched)
        self.create_subscription(String, '/surface_prediction', self.pred_cb, 10)

        # Publishers
        # We publish the modified global map subscribed from rtabmap. 
        self.pub_map = self.create_publisher(PointCloud2, '/rtabmap/semantic_map', qos_latched)
        
        # Timer for publishing map (1 Hz) to prevent network congestion
        self.create_timer(1.0, self.publish_map_loop)

        self.get_logger().info("Retroactive Painter Started. Waiting for Global Map...")

    def map_cb(self, msg):
        '''Update ground semantic labels incrementally from the global rgb map'''
        self.get_logger().info(f"Received new Global Map with {msg.width * msg.height} points.")
        
        old_points = self.global_points
        old_colors = self.global_colors
        old_tree   = self.kdtree

        gen = pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
        data = list(gen)
        
        if not data:
            return

        dtype_list = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<f4')]
        structured_data = np.array(data, dtype=dtype_list)

        x = structured_data['x']
        y = structured_data['y']
        z = structured_data['z']
        rgb = structured_data['rgb']
        
        # New points
        new_points = np.column_stack((x, y, z)).astype(np.float32)
        
        # New colors from RTAB-Map
        rgb_packed = structured_data['rgb']
        rgb_uint32 = rgb_packed.view(np.uint32)
        r = (rgb_uint32 >> 16) & 0xFF
        g = (rgb_uint32 >> 8) & 0xFF
        b = (rgb_uint32) & 0xFF
        new_colors = np.column_stack((r, g, b)).astype(np.float32)

        if old_points is not None and old_colors is not None and old_tree is not None:
            self.get_logger().info("Attempting to transfer old colors to new map...")
            
            # For every point in the new map, find the closest point in the old map
            dists, indices = old_tree.query(new_points, distance_upper_bound=0.05)
            
            valid_mask = dists != float('inf')
            
            # Where we found a match, copy the old color to the new map
            valid_indices_new = np.where(valid_mask)[0]
            valid_indices_old = indices[valid_mask]
            
            # Update new_colors with old_colors where matches occurred
            new_colors[valid_indices_new] = old_colors[valid_indices_old]

        self.global_points = new_points
        self.global_colors = new_colors
        
        self.kdtree = cKDTree(self.global_points)
        self.map_header = msg.header
        
        self.needs_republish = True
        self.get_logger().info("Map processed and colors transferred.")
        
    def pred_cb(self, msg):
        """Called frequently (Radar Rate)"""
        if self.kdtree is None or self.global_points is None:
            return

        try:
            pred_data = json.loads(msg.data)
            label = pred_data['predicted_label']
        except:
            return

        # TF (Global Map Frame -> Radar Frame)
        try:
            trans = self.tf_buffer.lookup_transform(
                self.radar_frame,           
                self.map_header.frame_id,   
                rclpy.time.Time())          
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            return

        # TF (Radar Frame -> Global Map Frame)
        # We need the inverse: Radar's position in Map
        try:
            # T_radar_to_map allows us to know where the robot is in the map
            trans_inv = self.tf_buffer.lookup_transform(
                 self.map_header.frame_id,
                 self.radar_frame,
                 rclpy.time.Time())
        except:
            return

        radar_pos_map = [
            trans_inv.transform.translation.x,
            trans_inv.transform.translation.y,
            trans_inv.transform.translation.z
        ]

        # Find map points close to radar (we don't process all points since it's super inefficient)
        indices = self.kdtree.query_ball_point(radar_pos_map, r=self.max_range)
        
        if len(indices) == 0:
            return

        # We only care about the subset of points found
        local_points_map = self.global_points[indices]        
        
        # Extract rotation/translation from T_map_to_radar
        q = trans.transform.rotation
        t = trans.transform.translation
        
        r_mat = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        t_vec = np.array([t.x, t.y, t.z])
        
        points_radar = np.dot(local_points_map, r_mat.T) + t_vec

        # Assume a camera pinhole model for the radar
        # Calculate angle from Z axis
        dists = np.linalg.norm(points_radar, axis=1)
        cos_theta = points_radar[:, 2] / (dists + 1e-6) 
        angles = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        
        # cone_mask = (np.abs(angles) < (self.fov_rad / 2.0)) & (points_radar[:, 2] > 0.0)
        cone_mask = (np.abs(angles) < (self.fov_rad / 2.0))
             
        valid_indices_subset = np.where(cone_mask)[0]
        
        if len(valid_indices_subset) == 0:
            return

        # Map back to global indices
        global_indices_to_paint = np.array(indices)[valid_indices_subset]

        target_color = COLOR_MAP.get(label, [128, 128, 128])
        
        # Direct assignment (Hard paint)
        # self.global_colors[global_indices_to_paint] = target_color
        
        # OR: Blending (Soft paint) - Smoother
        # New = 0.8 * Old + 0.2 * New
        old_colors = self.global_colors[global_indices_to_paint]
        new_colors = (old_colors * 0.5) + (np.array(target_color) * 0.5)
        self.global_colors[global_indices_to_paint] = new_colors

        self.needs_republish = True

    def publish_map_loop(self):
        """Periodically publish the entire updated map"""
        if not self.needs_republish or self.global_points is None:
            return

        # Re-pack the cloud
        # Pack RGB
        colors_u8 = self.global_colors.astype(np.uint8)
        rgb_int = (colors_u8[:, 0].astype(np.uint32) << 16) | \
                  (colors_u8[:, 1].astype(np.uint32) << 8)  | \
                  (colors_u8[:, 2].astype(np.uint32))
        
        rgb_packed = rgb_int.view(np.float32)
        
        final_data = np.column_stack((self.global_points, rgb_packed))

        # Create msg
        msg = pc2.create_cloud(
            self.map_header, # Use original frame and timestamp (or update timestamp)
            [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            ],
            final_data
        )
        
        self.pub_map.publish(msg)
        self.needs_republish = False

def main():
    rclpy.init()
    node = RadarSemanticSurfaceMapping()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()