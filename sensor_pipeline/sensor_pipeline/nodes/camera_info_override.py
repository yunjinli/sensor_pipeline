#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
import yaml
from pathlib import Path
# from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy,
    QoSHistoryPolicy, qos_profile_sensor_data
)
def _as_float_list(x):
    # flatten nested lists/tuples and cast to float
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            if isinstance(v, (list, tuple)):
                out.extend(_as_float_list(v))
            else:
                out.append(float(v))
        return out
    # handle scalars
    return [float(x)]

def load_info(path):
    d = yaml.safe_load(Path(path).read_text())

    # Accept both ROS YAML (camera_matrix/data) and OpenCV YAML (K,D,R,P)
    def get_mat(key_ros, key_cv, expected_len):
        if key_ros in d and 'data' in d[key_ros]:
            vals = _as_float_list(d[key_ros]['data'])
        elif key_cv in d:
            vals = _as_float_list(d[key_cv])
        else:
            raise KeyError(f'Missing {key_ros}.data or {key_cv} in {path}')
        if len(vals) != expected_len:
            raise ValueError(f'{key_ros or key_cv} length {len(vals)} != {expected_len}')
        return vals

    ci = CameraInfo()
    ci.width  = int(d.get('image_width', 0))
    ci.height = int(d.get('image_height', 0))
    ci.distortion_model = d.get('distortion_model', 'plumb_bob')

    # K(3x3)=9, D(5)=5 (adjust if your model has different length), R(3x3)=9, P(3x4)=12
    ci.k = get_mat('camera_matrix', 'K', 9)
    # distortion can appear as ROS data or as D
    if 'distortion_coefficients' in d and 'data' in d['distortion_coefficients']:
        ci.d = _as_float_list(d['distortion_coefficients']['data'])
    elif 'D' in d:
        ci.d = _as_float_list(d['D'])
    else:
        ci.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    # common is 5; adjust if you have more terms
    if len(ci.d) not in (4,5,8,12,14):  # allow common sizes
        raise ValueError(f'D length {len(ci.d)} looks unusual; check your YAML')

    # R and P
    try:
        ci.r = get_mat('rectification_matrix', 'R', 9)
    except Exception:
        ci.r = [1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0]
    try:
        ci.p = get_mat('projection_matrix', 'P', 12)
    except Exception:
        # default pinhole P from K
        fx, fy, cx, cy = ci.k[0], ci.k[4], ci.k[2], ci.k[5]
        ci.p = [fx,0.0,cx,0.0,  0.0,fy,cy,0.0,  0.0,0.0,1.0,0.0]

    return ci

class CameraInfoOverride(Node):
    def __init__(self):
        super().__init__('camera_info_override')
        self.declare_parameter('image_topic', '')
        self.declare_parameter('output_camera_info_topic', '')
        self.declare_parameter('calib_yaml', '')
        self.declare_parameter('frame_id', '')

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.info_topic  = self.get_parameter('output_camera_info_topic').get_parameter_value().string_value
        self.frame_id    = self.get_parameter('frame_id').get_parameter_value().string_value
        yaml_path        = self.get_parameter('calib_yaml').get_parameter_value().string_value
        self.base_info   = load_info(yaml_path)

        # self.sub = self.create_subscription(Image, self.image_topic, self.on_image, 10)
        # self.pub = self.create_publisher(CameraInfo, self.info_topic, 10)
        self.declare_parameter('durability', 'transient_local')  # or 'volatile'
        durability = self.get_parameter('durability').get_parameter_value().string_value

        # subscriber for images (unchanged):
        self.sub = self.create_subscription(Image, self.image_topic, self.on_image, qos_profile_sensor_data)

        # build publisher QoS from param
        dur = QoSDurabilityPolicy.TRANSIENT_LOCAL if durability == 'transient_local' else QoSDurabilityPolicy.VOLATILE
        info_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=dur,
        )
        self.pub = self.create_publisher(CameraInfo, self.info_topic, info_qos)
        self.get_logger().info(f'Overriding CameraInfo on {self.info_topic} from {yaml_path}')

    def on_image(self, img: Image):
        ci = CameraInfo()
        # self.get_logger().info(f'Received image (h, w): {img.height}, {img.width}')
        ci.header = img.header
        if self.frame_id:
            ci.header.frame_id = self.frame_id
        
        if self.base_info.width != img.width or self.base_info.height != img.height:
            self.get_logger().info(f'Image size ({img.width}x{img.height}) does not match calib ({self.base_info.width}x{self.base_info.height}); adjusting CameraInfo accordingly.')
            new_k = [self.base_info.k[0] / self.base_info.width * img.width, 0.0, self.base_info.k[2] / self.base_info.width * img.width,
                        0.0, self.base_info.k[4] / self.base_info.height * img.height, self.base_info.k[5] / self.base_info.height * img.height,
                        0.0, 0.0, 1.0]
            self.base_info.k = new_k
            self.base_info.p = [new_k[0], 0.0, new_k[2], 0.0,
                                0.0, new_k[4], new_k[5], 0.0,
                                0.0, 0.0, 1.0, 0.0]
            
            self.base_info.width = img.width
            self.base_info.height = img.height
            
            
        ci.width  = self.base_info.width
        ci.height = self.base_info.height
        ci.distortion_model = self.base_info.distortion_model
        ci.d = list(self.base_info.d)
        ci.k = list(self.base_info.k)
        ci.r = list(self.base_info.r)
        ci.p = list(self.base_info.p)
        self.pub.publish(ci)

def main():
    rclpy.init()
    rclpy.spin(CameraInfoOverride())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
