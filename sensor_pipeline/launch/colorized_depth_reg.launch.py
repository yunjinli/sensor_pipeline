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
    share = get_package_share_directory('sensor_pipeline')
    rgb_yaml = os.path.join(share, 'config', 'realsense_calib.yaml')
    tof_yaml = os.path.join(share, 'config', 'pmd_calib.yaml')
    tf_cam1_cam0_yaml = os.path.join(share, 'config', 'tf_cam1_cam0.yaml')

    # rgb_undist_yaml = os.path.join(share, 'config', 'realsense_calib_undist.yaml')
    # tof_undist_yaml = os.path.join(share, 'config', 'pmd_calib_undist.yaml')
    
    # pmd_royale_ros_examples_path = get_package_share_directory('pmd_royale_ros_examples')
    flexx_config = os.path.join(share, 'config', 'flexx2.yaml')
    
    # --- If instead your matrix is T_tof<-rgb (pose of RGB in TOF frame), use this one: ---
    # static_tf_rgb_in_tof = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     arguments=[
    #         '-0.02820566', '0.05938304', '-0.00502076',
    #         '0.00154171', '-0.00451936', '-0.70765389', '0.70654311',
    #         'camera_color_optical_frame', 'pmd_royale_ros_camera_node_optical_frame'
    #     ],
    #     name='rgb_in_tof'
    # )

    Rm_cam1_cam0, t_cam1_cam0 = load_tf_yaml(tf_cam1_cam0_yaml)
    Rm_cam0_cam1 = Rm_cam1_cam0.T
    t_cam0_cam1 = -Rm_cam0_cam1 @ t_cam1_cam0
    related_pose_cam0_cam1 = rt_to_pose(Rm=Rm_cam0_cam1, t=t_cam0_cam1)
    # related_pose_txtytzqxqyqzqw = f"{related_pose_cam0_cam1.position.x} {related_pose_cam0_cam1.position.y} {related_pose_cam0_cam1.position.z} {related_pose_cam0_cam1.orientation.x} {related_pose_cam0_cam1.orientation.y} {related_pose_cam0_cam1.orientation.z} {related_pose_cam0_cam1.orientation.w}"
    # print(related_pose_txtytzqxqyqzqw)
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
    # --- RealSense (RGB only) ---
    realsense_rgb = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='realsense',
        output='screen',
        parameters=[{
            'enable_color': True,
            'enable_depth': False,              # we colorize with ToF depth
            'rgb_camera.enable_auto_exposure': True
        }]
        # Frame will be camera_color_optical_frame by default
    )
    
    tof_depth = ComposableNodeContainer(
        name='pmd_royale_ros_camera_node_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='pmd_royale_ros_driver',
                plugin='pmd_royale_ros_driver::CameraNode',
                name='pmd_royale_ros_camera_node',
                parameters=[flexx_config]
            )
        ],
        output='screen',
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
            # Rectify ToF depth
            # ComposableNode(
            #     package='image_proc', plugin='image_proc::RectifyNode', name='tof_rectify',
            #     remappings=[
            #         ('image', '/pmd_royale_ros_camera_node/gray_image_0'),
            #         ('camera_info', '/pmd_royale_ros_camera_node/camera_info_over_tl'),
            #         ('image_rect', '/tof/gray_image_rect')
            #     ]
            # ),
            # Rectify ToF depth
            ComposableNode(
                package='image_proc', plugin='image_proc::RectifyNode', name='tof_rectify',
                remappings=[
                    ('image', '/pmd_royale_ros_camera_node/depth_image_0'),
                    ('camera_info', '/pmd_royale_ros_camera_node/camera_info_over_tl'),
                    ('image_rect', '/tof/depth_image_rect')
                ]
            ),
            # Register ToF depth into RGB model (needs VOLATILE camera_info)
            ComposableNode(
                package='depth_image_proc', plugin='depth_image_proc::RegisterNode', name='register_depth_rgb',
                remappings=[
                    ('depth/image_rect',  '/tof/depth_image_rect'),
                    ('depth/camera_info', '/pmd_royale_ros_camera_node/camera_info_over'),   # VOLATILE
                    ('rgb/camera_info',       '/camera/realsense/color/camera_info_over'),       # VOLATILE
                    ('depth_registered/image_rect',  '/rgb/depth_registered'),
                    ('depth_registered/camera_info', '/camera/realsense/color/camera_info_over')
                ],
                parameters=[{
                    'approximate_sync': True,
                    'queue_size': 50
                }]
            ),

            # Make colored point cloud in RGB frame
            ComposableNode(
                package='depth_image_proc', plugin='depth_image_proc::PointCloudXyzrgbNode', name='cloud_xyzrgb',
                remappings=[
                    ('rgb/image_rect_color',        '/rgb/image_rect_color'),
                    ('rgb/camera_info',             '/camera/realsense/color/camera_info_over'),  # VOLATILE
                    ('depth_registered/image_rect', '/rgb/depth_registered'),
                    ('points',                      '/rgb/points_xyzrgb')
                ],
                parameters=[{
                    'approximate_sync': True,
                    'queue_size': 50
                }]
            ),
        ]
    )
    
    # colorized_pcd = Node(
    #     package='sensor_pipeline',
    #     executable='colorize_pointcloud',
    #     name='colorize_pointcloud',
    #     output='screen',
    #     parameters=[{
    #         'pc_topic': '/pmd_royale_ros_camera_node/point_cloud_0',
    #         'rgb_image_topic': '/rgb/image_rect_color',
    #         'rgb_camera_info_topic': '/camera/realsense/color/camera_info_over',
    #         'rgb_optical_frame': 'camera_color_optical_frame',
    #         'output_topic': '/rgb/points_xyzrgb',
    #         'queue_size': 20,
    #     }]
    # )

    return LaunchDescription([static_tf_tof_in_rgb, realsense_rgb, tof_depth,
    # return LaunchDescription([static_tf_tof_in_rgb, tof_depth,
                              rgb_info_tl, rgb_info_v, tof_info_tl, tof_info_v,
                              container])
