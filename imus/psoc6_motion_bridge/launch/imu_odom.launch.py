from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # ---- Launch-time arguments (so you can override without editing the file)
    port_arg = DeclareLaunchArgument("port", default_value="/dev/ttyACM0")
    baud_arg = DeclareLaunchArgument("baud", default_value="230400")
    frame_id_arg = DeclareLaunchArgument("frame_id", default_value="imu_link")
    raw_topic_arg = DeclareLaunchArgument("raw_topic", default_value="/imu/data")            # output of bridge
    oriented_topic_arg = DeclareLaunchArgument("oriented_topic", default_value="/imu/data_oriented")

    port = LaunchConfiguration("port")
    baud = LaunchConfiguration("baud")
    frame_id = LaunchConfiguration("frame_id")
    raw_topic = LaunchConfiguration("raw_topic")
    oriented_topic = LaunchConfiguration("oriented_topic")

    # ---- psoc6_motion_bridge node (publishes IMU raw data)
    psoc6_bridge = Node(
        package="psoc6_motion_bridge",
        executable="psoc6_motion_bridge",
        name="psoc6_motion_bridge",
        output="screen",
        parameters=[{
            "port": port,
            "baud": baud,
            "frame_id": frame_id,
            "topic": raw_topic,        # this node lets you choose the pub topic via a param
        }],
    )

    # ---- imu_filter_madgwick node (subscribes raw, publishes oriented)
    madgwick_filter = Node(
        package="imu_filter_madgwick",
        executable="imu_filter_madgwick_node",
        name="imu_filter_madgwick",
        output="screen",
        parameters=[{
            "use_mag": False,
            "world_frame": "enu",
        }],
        # remaps: raw input to whatever the bridge is publishing; oriented output to target topic
        remappings=[
            ("imu/data_raw", raw_topic),
            ("imu/data", oriented_topic),
        ],
    )

    return LaunchDescription([
        port_arg, baud_arg, frame_id_arg, raw_topic_arg, oriented_topic_arg,
        psoc6_bridge,
        madgwick_filter,
    ])
