# parent.launch.py
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution as P
from ament_index_python.packages import get_package_share_directory
import os
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch.conditions import IfCondition

def generate_launch_description():
    share = get_package_share_directory('sensor_pipeline')
    imu_pkg = FindPackageShare('psoc6_motion_bridge')
    radar_pkg = FindPackageShare('bgt60tr13c_driver')
    flexx_config = os.path.join(share, 'config', 'flexx2.yaml')
    
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

    imu = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(P([imu_pkg, 'launch', 'imu_odom.launch.py'])),
        launch_arguments={'port': '/dev/imu'}.items()
    )

    # radar_vis_arg = DeclareLaunchArgument(
    #     'radar_vis',
    #     default_value='true',   # or 'true' if you want it on by default
    #     description='Run the visualization node (true/false)'
    # )
    # radar_vis = LaunchConfiguration('radar_vis')

    radar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(P([radar_pkg, 'launch', 'radar.launch.py'])),
        launch_arguments={'port': '/dev/radar'}.items()
    )

    rviz_config = os.path.join(share, 'config', 'sensors_display.rviz')
    run_rviz_arg = DeclareLaunchArgument(
        'run_rviz',
        default_value='true',   # or 'true' if you want it on by default
        description='Run the visualization in Rviz'
    )
    run_rviz = LaunchConfiguration('run_rviz')

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],   # <- use your saved config
        output='screen',
        condition=IfCondition(run_rviz),
    )


    return LaunchDescription([imu, radar, realsense_rgb, tof_depth, run_rviz_arg, rviz])
