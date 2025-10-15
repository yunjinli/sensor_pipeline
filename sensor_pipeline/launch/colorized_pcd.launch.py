from launch import LaunchDescription
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    container = ComposableNodeContainer(
        name='fusion_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        output='screen',
        composable_node_descriptions=[
            # 1) Register ToF depth into RGB camera model
            ComposableNode(
                package='depth_image_proc',
                plugin='depth_image_proc::RegisterNode',
                name='register_depth',
                remappings=[
                    # Source ToF (rectified) + its CameraInfo (overridden)
                    ('depth/image_rect',  '/tof/depth_image_rect'),
                    ('depth/camera_info', '/pmd_royale_ros_camera_node/camera_info_over'),
                    # Target RGB CameraInfo (overridden)
                    ('camera_info',       '/camera/realsense/color/camera_info_over'),
                    # Output (depth in RGB model)
                    ('depth_registered/image_rect',  '/rgb/depth_registered'),
                    ('depth_registered/camera_info', '/camera/realsense/color/camera_info_over')
                ]
            ),
            # 2) Colorize into a point cloud (in the RGB frame)
            ComposableNode(
                package='depth_image_proc',
                plugin='depth_image_proc::PointCloudXyzrgbNode',
                name='cloud_xyzrgb',
                remappings=[
                    ('rgb/image_rect_color',        '/rgb/image_rect_color'),
                    ('rgb/camera_info',             '/camera/realsense/color/camera_info_over'),
                    ('depth_registered/image_rect', '/rgb/depth_registered'),
                    ('points',                      '/rgb/points_xyzrgb')
                ]
            ),
        ]
    )

    return LaunchDescription([container])
