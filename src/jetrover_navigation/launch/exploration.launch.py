"""Start online exploration and HOME manager; optionally include common navigation."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("jetrover_navigation")
    return LaunchDescription(
        [
            DeclareLaunchArgument("start_navigation", default_value="true"),
            DeclareLaunchArgument("auto_start", default_value="true"),
            DeclareLaunchArgument(
                "backend", default_value="upstream", choices=["upstream", "custom"]
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=os.path.join(share, "config", "exploration_isaac.yaml"),
            ),
            DeclareLaunchArgument(
                "upstream_params",
                default_value=os.path.join(share, "config", "frontier_ros2_isaac.yaml"),
            ),
            DeclareLaunchArgument("initial_spin", default_value="false"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("map_frame", default_value="map"),
            DeclareLaunchArgument(
                "nav2_params",
                default_value=os.path.join(share, "config", "nav2_isaac.yaml"),
            ),
            DeclareLaunchArgument(
                "slam_params",
                default_value=os.path.join(share, "config", "isaac_slam.yaml"),
            ),
            DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(share, "launch", "navigation.launch.py")
                ),
                launch_arguments={
                    key: LaunchConfiguration(key)
                    for key in (
                        "use_sim_time",
                        "nav2_params",
                        "slam_params",
                        "cmd_vel_topic",
                    )
                }.items(),
                condition=IfCondition(LaunchConfiguration("start_navigation")),
            ),
            Node(
                package="jetrover_navigation",
                executable="exploration_manager",
                name="exploration_manager",
                output="screen",
                condition=IfCondition(
                    PythonExpression(
                        ["'", LaunchConfiguration("backend"), "' == 'custom'"]
                    )
                ),
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "auto_start": ParameterValue(
                            LaunchConfiguration("auto_start"), value_type=bool
                        ),
                        "use_sim_time": ParameterValue(
                            LaunchConfiguration("use_sim_time"), value_type=bool
                        ),
                        "base_frame": LaunchConfiguration("base_frame"),
                        "map_frame": LaunchConfiguration("map_frame"),
                    },
                ],
            ),
            Node(
                package="frontier_exploration_ros2",
                executable="frontier_explorer",
                name="frontier_explorer",
                output="screen",
                condition=IfCondition(
                    PythonExpression(
                        ["'", LaunchConfiguration("backend"), "' == 'upstream'"]
                    )
                ),
                parameters=[
                    LaunchConfiguration("upstream_params"),
                    {
                        "use_sim_time": ParameterValue(
                            LaunchConfiguration("use_sim_time"), value_type=bool
                        ),
                        "robot_base_frame": LaunchConfiguration("base_frame"),
                        "global_frame": LaunchConfiguration("map_frame"),
                        "autostart": False,
                        "control_service_enabled": True,
                    },
                ],
            ),
            Node(
                package="jetrover_navigation",
                executable="frontier_bridge",
                name="frontier_bridge",
                output="screen",
                condition=IfCondition(
                    PythonExpression(
                        ["'", LaunchConfiguration("backend"), "' == 'upstream'"]
                    )
                ),
                parameters=[
                    {
                        "use_sim_time": ParameterValue(
                            LaunchConfiguration("use_sim_time"), value_type=bool
                        ),
                        "base_frame": LaunchConfiguration("base_frame"),
                        "map_frame": LaunchConfiguration("map_frame"),
                        "auto_start": ParameterValue(
                            LaunchConfiguration("auto_start"), value_type=bool
                        ),
                        "initial_spin": ParameterValue(
                            LaunchConfiguration("initial_spin"), value_type=bool
                        ),
                    }
                ],
            ),
        ]
    )
