from launch import LaunchDescription
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory
import os
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
from scipy.spatial.transform import Rotation as R
import yaml
from pathlib import Path
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

# from launch.substitutions import PathJoinSubstitution
# from launch_ros.substitutions import FindPackageShare

def load_tf_yaml(path):
    d = yaml.safe_load(Path(path).read_text())
    t = np.array(d['T']).reshape(3)
    Rm = np.array(d['R']).reshape(3,3)
    return Rm, t
    
def rt_to_pose(Rm: np.ndarray, t: np.ndarray) -> Pose:
    # Rm: (3,3), t: (3,)
    # Orthonormalize (optional but recommended)
    U, _, Vt = np.linalg.svd(Rm)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, 2] *= -1
        R_ortho = U @ Vt

    qx, qy, qz, qw = R.from_matrix(R_ortho).as_quat()  # returns (x, y, z, w)
    p = Pose()
    p.position.x, p.position.y, p.position.z = float(t[0]), float(t[1]), float(t[2])
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = qx, qy, qz, qw
    return p

def generate_launch_description():
    run_rviz_arg = DeclareLaunchArgument(
        'run_rviz',             # Name of the argument
        default_value='false',   # Default to true (or 'false' if you prefer)
        description='Whether to launch RViz'
    )
    
    share = get_package_share_directory('sensor_pipeline')
    rgb_yaml = os.path.join(share, 'config', 'realsense_calib.yaml')
    tof_yaml = os.path.join(share, 'config', 'pmd_calib.yaml')
    tf_cam1_cam0_yaml = os.path.join(share, 'config', 'tf_cam1_cam0.yaml')
    
    Rm_cam1_cam0, t_cam1_cam0 = load_tf_yaml(tf_cam1_cam0_yaml)
    Rm_cam0_cam1 = Rm_cam1_cam0.T
    t_cam0_cam1 = -Rm_cam0_cam1 @ t_cam1_cam0
    related_pose_cam0_cam1 = rt_to_pose(Rm=Rm_cam0_cam1, t=t_cam0_cam1)
    
    static_tf_tof_in_rgb = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        # (x y z qx qy qz qw parent child)
        arguments=[
            # T_rgb<-tof  (pose of TOF in RGB frame)
            f'{related_pose_cam0_cam1.position.x}', f'{related_pose_cam0_cam1.position.y}', f'{related_pose_cam0_cam1.position.z}',
            f'{related_pose_cam0_cam1.orientation.x}', f'{related_pose_cam0_cam1.orientation.y}', f'{related_pose_cam0_cam1.orientation.z}', f'{related_pose_cam0_cam1.orientation.w}',
            # '-0.02820566', '0.05938304', '-0.00502076',
            # '0.00154171', '-0.00451936', '-0.70765389', '0.70654311',
            'camera_color_optical_frame', 'pmd_royale_ros_camera_node_optical_frame'
        ],
        name='tof_in_rgb'
    )
    
    Rm_cam0_imu = np.array([[ 0, -1,  0],
                            [ 0,  0, -1],
                            [1,  0,  0]])
    t_cam0_imu = np.array([0, 0.02, -0.05])
    
    related_pose_cam0_imu = rt_to_pose(Rm=Rm_cam0_imu, t=t_cam0_imu)
    
    static_tf_imu_in_rgb = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            f'{related_pose_cam0_imu.position.x}', f'{related_pose_cam0_imu.position.y}', f'{related_pose_cam0_imu.position.z}',
            f'{related_pose_cam0_imu.orientation.x}', f'{related_pose_cam0_imu.orientation.y}', f'{related_pose_cam0_imu.orientation.z}', f'{related_pose_cam0_imu.orientation.w}',
            'camera_color_optical_frame', 'imu_link'
            # 'imu_link', 'camera_color_optical_frame'
        ]
    )
    
    # --- CameraInfo overrides (two per camera: TL for Rectify, VOLATILE for Register/Cloud) ---
    rgb_info_tl = Node(
        package='sensor_pipeline', executable='camera_info_override', name='rgb_info_tl',
        parameters=[{
            'image_topic': '/camera/realsense/color/image_raw',
            # 'image_topic': '/camera/camera/color/image_raw',
            'output_camera_info_topic': '/camera/realsense/color/camera_info_over_tl',
            'calib_yaml': rgb_yaml, 'frame_id': 'camera_color_optical_frame',
            'durability': 'transient_local',
        }]
    )
    rgb_info_v = Node(
        package='sensor_pipeline', executable='camera_info_override', name='rgb_info_v',
        parameters=[{
            'image_topic': '/camera/realsense/color/image_raw',
            # 'image_topic': '/camera/camera/color/image_raw',
            'output_camera_info_topic': '/camera/realsense/color/camera_info_over',
            'calib_yaml': rgb_yaml, 'frame_id': 'camera_color_optical_frame',
            'durability': 'volatile',
        }]
    )
    tof_info_tl = Node(
        package='sensor_pipeline', executable='camera_info_override', name='tof_info_tl',
        parameters=[{
            'image_topic': '/pmd_royale_ros_camera_node/depth_image_0',
            'output_camera_info_topic': '/pmd_royale_ros_camera_node/camera_info_over_tl',
            'calib_yaml': tof_yaml, 'frame_id': 'pmd_royale_ros_camera_node_optical_frame',
            'durability': 'transient_local',
        }]
    )
    tof_info_v = Node(
        package='sensor_pipeline', executable='camera_info_override', name='tof_info_v',
        parameters=[{
            'image_topic': '/pmd_royale_ros_camera_node/depth_image_0',
            'output_camera_info_topic': '/pmd_royale_ros_camera_node/camera_info_over',
            'calib_yaml': tof_yaml, 'frame_id': 'pmd_royale_ros_camera_node_optical_frame',
            'durability': 'volatile',
        }]
    )

    # --- Processing: Rectify (uses TL CameraInfo), Register, PointCloud (use VOLATILE) ---
    container = ComposableNodeContainer(
        name='fusion_container', namespace='',
        package='rclcpp_components', executable='component_container_mt', output='screen',
        composable_node_descriptions=[
            # Rectify RGB
            ComposableNode(
                package='image_proc', plugin='image_proc::RectifyNode', name='rgb_rectify',
                remappings=[
                    ('image', '/camera/realsense/color/image_raw'),
                    ('camera_info', '/camera/realsense/color/camera_info_over_tl'),
                    ('image_rect', '/rgb/image_rect_color')
                ]
            ),
        ]
    )
    
    colorized_pcd = Node(
        package='sensor_pipeline',
        executable='depth_overlay',
        name='depth_overlay',
        output='screen',
        parameters=[{
            'pc_topic': '/pmd_royale_ros_camera_node/point_cloud_0',
            'rgb_image_topic': '/rgb/image_rect_color',
            'rgb_camera_info_topic': '/camera/realsense/color/camera_info_over',
            'rgb_optical_frame': 'camera_color_optical_frame',
            'queue_size': 20,
        }]
    )
    
    rviz_config = os.path.join(share, 'config', 'rgbtof.rviz')

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],   # <- use your saved config
        output='screen',
        condition=IfCondition(LaunchConfiguration('run_rviz'))
    )

    return LaunchDescription([run_rviz_arg, static_tf_tof_in_rgb, static_tf_imu_in_rgb,
                              rgb_info_tl, rgb_info_v, tof_info_tl, tof_info_v,
                              container, colorized_pcd, rviz])
