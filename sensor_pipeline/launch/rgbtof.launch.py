# ****************************************************************************\
# * Copyright (C) 2023 pmdtechnologies ag
# *
# * THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
# * KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# * IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
# * PARTICULAR PURPOSE.
# *
# ****************************************************************************/

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory
from os.path import join

def generate_launch_description():
    share = get_package_share_directory('pmd_royale_ros_examples')
    flexx_config = join(share, 'config', 'flexx2.yaml')

    # static_tf_tof_in_rgb = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     # (x y z qx qy qz qw parent child)
    #     arguments=[
    #         # T_rgb<-tof  (pose of TOF in RGB frame)
    #         '0.05935873', '0.02833985', '0.00452840',
    #         '-0.00154171', '0.00451936', '0.70765389', '0.70654311',
    #         'camera_color_optical_frame', 'pmd_royale_ros_camera_node_optical_frame'
    #     ],
    #     name='tof_in_rgb'
    # )
    
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

    return LaunchDescription([tof_depth, realsense_rgb])

