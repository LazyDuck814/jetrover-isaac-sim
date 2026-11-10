"""Isaac Sim online SLAM + Nav2. No manufacturer launch dependencies."""

import os
import json

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    RegisterEventHandler,
    LogInfo,
)
from launch.events import matches_action
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch_ros.parameter_descriptions import ParameterValue
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    share = get_package_share_directory("jetrover_navigation")
    # Official Jazzy data ships with nav2_smac_planner; no generated/custom driver.
    lattice = os.path.join(
        get_package_share_directory("nav2_smac_planner"),
        "sample_primitives",
        "5cm_resolution",
        "0.5m_turning_radius",
        "diff",
        "output.json",
    )
    with open(lattice, encoding="utf-8") as stream:
        metadata = json.load(stream)["lattice_metadata"]
    if metadata["motion_model"] != "diff" or metadata["grid_resolution"] != 0.05:
        raise ValueError(
            "Expected Nav2 differential-drive primitives at 0.05 m resolution"
        )
    sim = {
        "use_sim_time": ParameterValue(
            LaunchConfiguration("use_sim_time"), value_type=bool
        )
    }
    params = LaunchConfiguration("nav2_params")
    slam = LifecycleNode(
        package="slam_toolbox",
        executable="sync_slam_toolbox_node",
        name="slam_toolbox",
        namespace="",
        output="screen",
        remappings=[("/map", "/navigation/slam_map")],
        parameters=[
            LaunchConfiguration("slam_params"),
            sim,
            {"mode": "mapping", "use_lifecycle_manager": False},
        ],
    )
    activate = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=slam,
            start_state="configuring",
            goal_state="inactive",
            entities=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(slam),
                        transition_id=Transition.TRANSITION_ACTIVATE,
                    )
                )
            ],
        )
    )
    configure = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(slam),
            transition_id=Transition.TRANSITION_CONFIGURE,
        )
    )
    # Every velocity producer goes through the SAME smoother. The USD expects Twist.
    servers = [
        (
            "nav2_controller",
            "controller_server",
            [("cmd_vel", "/navigation/cmd_vel_raw")],
        ),
        ("nav2_planner", "planner_server", []),
        ("nav2_smoother", "smoother_server", []),
        ("nav2_behaviors", "behavior_server", [("cmd_vel", "/navigation/cmd_vel_raw")]),
        ("nav2_bt_navigator", "bt_navigator", []),
        (
            "nav2_velocity_smoother",
            "velocity_smoother",
            [
                ("cmd_vel", "/navigation/cmd_vel_raw"),
                ("cmd_vel_smoothed", LaunchConfiguration("cmd_vel_topic")),
            ],
        ),
    ]
    nodes = [
        Node(
            package=pkg,
            executable=name,
            name=name,
            output="screen",
            parameters=[params, sim]
            + (
                [{"default_nav_to_pose_bt_xml": LaunchConfiguration("bt_xml")}]
                if name == "bt_navigator"
                else []
            )
            + (
                [{"GridBased.lattice_filepath": lattice}]
                if name == "planner_server"
                else []
            ),
            remappings=remaps,
        )
        for pkg, name, remaps in servers
    ]
    manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_navigation",
        output="screen",
        parameters=[
            sim,
            {
                "autostart": ParameterValue(
                    LaunchConfiguration("autostart"), value_type=bool
                ),
                "node_names": [name for _, name, _ in servers],
            },
        ],
    )
    scan = Node(
        package="jetrover_navigation",
        executable="scan_filter",
        name="jetrover_scan_filter",
        output="screen",
        prefix=["/usr/bin/python3"],
        condition=IfCondition(LaunchConfiguration("start_scan_filter")),
        parameters=[LaunchConfiguration("scan_filter_params"), sim],
        remappings=[
            ("scan_raw", LaunchConfiguration("scan_raw_topic")),
            ("scan", LaunchConfiguration("scan_topic")),
        ],
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "bt_xml",
                default_value=os.path.join(share, "config", "exploration_nav.xml"),
            ),
            DeclareLaunchArgument(
                "nav2_params",
                default_value=os.path.join(share, "config", "nav2_isaac.yaml"),
            ),
            DeclareLaunchArgument(
                "slam_params",
                default_value=os.path.join(share, "config", "isaac_slam.yaml"),
            ),
            DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("start_scan_filter", default_value="true"),
            DeclareLaunchArgument("scan_raw_topic", default_value="/scan_raw"),
            DeclareLaunchArgument("scan_topic", default_value="/scan"),
            DeclareLaunchArgument(
                "scan_filter_params",
                default_value=os.path.join(share, "config", "lidar_filters.yaml"),
            ),
            DeclareLaunchArgument("autostart", default_value="true"),
            LogInfo(
                msg=[
                    "[NAV PROFILE forward-diff-v1] params=",
                    params,
                    " | differential lattice=",
                    lattice,
                    " | no reverse recovery | standard Nav2 collision checking",
                ]
            ),
            activate,
            scan,
            slam,
            configure,
            Node(
                package="jetrover_navigation",
                executable="map_padding",
                prefix=["/usr/bin/python3"],
                output="screen",
                parameters=[sim, {"margin_m": 0.6}],
            ),
            *nodes,
            manager,
        ]
    )
