from launch import LaunchDescription
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    share = get_package_share_directory('sensor_pipeline')
    rgb_yaml = os.path.join(share, 'config', 'realsense_calib.yaml')
    tof_yaml = os.path.join(share, 'config', 'pmd_calib.yaml')

    # --- Override CameraInfo (publish *_over topics using your YAMLs) ---
    rgb_info = Node(
        package='sensor_pipeline', executable='camera_info_override', name='rgb_info_override',
        parameters=[{
            'image_topic': '/camera/realsense/color/image_raw',
            'output_camera_info_topic': '/camera/realsense/color/camera_info_over',
            'calib_yaml': rgb_yaml,
            'frame_id': 'camera_color_optical_frame'
        }]
    )
    tof_info = Node(
        package='sensor_pipeline', executable='camera_info_override', name='tof_info_override',
        parameters=[{
            'image_topic': '/pmd_royale_ros_camera_node/depth_image_0',
            'output_camera_info_topic': '/pmd_royale_ros_camera_node/camera_info_over',
            'calib_yaml': tof_yaml,
            'frame_id': 'pmd_royale_ros_camera_node_optical_frame'
        }]
    )

    # --- Rectify using the OVERRIDDEN camera_info topics ---
    proc = ComposableNodeContainer(
        name='rectify_container', namespace='',
        package='rclcpp_components', executable='component_container_mt', output='screen',
        composable_node_descriptions=[
            ComposableNode(
                package='image_proc', plugin='image_proc::RectifyNode', name='rgb_rectify',
                remappings=[
                    ('image',       '/camera/realsense/color/image_raw'),
                    ('camera_info', '/camera/realsense/color/camera_info_over'),
                    ('image_rect',  '/rgb/image_rect_color')
                ]
            ),
            ComposableNode(
                package='image_proc', plugin='image_proc::RectifyNode', name='tof_rectify',
                remappings=[
                    ('image',       '/pmd_royale_ros_camera_node/depth_image_0'),
                    ('camera_info', '/pmd_royale_ros_camera_node/camera_info_over'),
                    ('image_rect',  '/tof/depth_image_rect')
                ]
            ),
        ]
    )

    return LaunchDescription([rgb_info, tof_info, proc])
