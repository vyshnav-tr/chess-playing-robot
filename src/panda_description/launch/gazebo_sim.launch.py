import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable
from launch.actions import ExecuteProcess
from launch.actions import TimerAction



def generate_launch_description():
    pkg_share = get_package_share_directory('ros_arm_panda_description')
    urdf_path = os.path.join(pkg_share, 'urdf', 'panda.urdf')  # adjust filename if different

    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description,'use_sim_time': True}]
    )
    gz_resource_path = SetEnvironmentVariable(
    name='GZ_SIM_RESOURCE_PATH',
    value=os.path.dirname(pkg_share) + ':' + os.path.join(pkg_share, 'worlds', 'models')
)

    gazebo = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
    ),
    launch_arguments={
        'gz_args': f'-r {os.path.join(pkg_share, "worlds", "pick_place_world.sdf")}'
    }.items()
)
    table_spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-file', os.path.join(pkg_share, 'worlds', 'models', 'table', 'model.sdf'),
            '-name', 'table',
            '-x', '0.42', '-y', '0.0', '-z', '0.0'
        ],
        output='screen'
    )
    chessboard_spawn = Node(
    package='ros_gz_sim',
    executable='create',
    arguments=[
        '-file', os.path.join(pkg_share, 'worlds', 'models', 'chessboard', 'model.sdf'),
        '-name', 'chessboard',
        '-x', '0.42', '-y', '0.0', '-z', '0.4'  # z = table height, so it sits on top
    ],
    output='screen'
)
    camera_spawn = Node(
    package='ros_gz_sim',
    executable='create',
    arguments=[
        '-file', os.path.join(pkg_share, 'worlds', 'models', 'overhead_camera', 'model.sdf'),
        '-name', 'overhead_camera',
        '-x', '0.42', '-y', '0.0', '-z', '1.2', '-R', '0.0', '-P', '1.5708', '-Y', '0.0'  # Adjust rotation as needed
    ],
    output='screen'
)
    camera_tf = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    arguments=[
        '--x', '0.42', '--y', '0', '--z', '1.2',
        '--roll', '3.14159', '--pitch', '0', '--yaw', '0',
        '--frame-id', 'panda_link0',
        '--child-frame-id', 'overhead_camera/camera_link/overhead_depth_cam'
    ],
    output='screen'
)
    camera_bridge = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    arguments=[
        '/overhead_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
        '/overhead_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
        '/overhead_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
        '/overhead_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
    ],
    output='screen'
)
    board_calibration_service = ExecuteProcess(
    cmd=['python3', os.path.join(
        get_package_share_directory('panda_moveit_config'),
        'scripts', 'board_calibration_service.py'
    )],
    output='screen'
)
    delayed_calibration = TimerAction(
    period=8.0,
    actions=[board_calibration_service]
)

    # --- Full chess set spawn ---
    # Board math matches the C++ pick/place node: BOARD_X/Y=0.42,0.0, SQUARE_SIZE=0.4/8
    BOARD_X = 0.42
    BOARD_Y = 0.0
    SQUARE_SIZE = 0.4 / 8.0
    PIECE_Z = 0.43  # top of board surface (board top at 0.42) + small clearance

    def square_xy(col, row):
        # col (a-h, the "file") runs left-right relative to the robot -> Y
        # row (the "rank"/side) runs toward/away from the robot -> X
        x = BOARD_X + (row - 3.5) * SQUARE_SIZE
        y = BOARD_Y + (col - 3.5) * SQUARE_SIZE
        return x, y

    COLS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    BACK_RANK_ORDER = ['rook', 'knight', 'bishop', 'king', 'queen', 'bishop', 'knight', 'rook']

    def make_piece_spawn(prefix, piece_type, col_idx, row_idx):
        x, y = square_xy(col_idx, row_idx)
        model_dir = f'{prefix}_{piece_type}'  # e.g. w_pawn, b_rook
        name = f'{prefix}_{piece_type}_{COLS[col_idx]}{row_idx + 1}'  # e.g. w_pawn_a2
        return Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-file', os.path.join(pkg_share, 'worlds', 'models', model_dir, 'model.sdf'),
                '-name', name,
                '-x', str(x), '-y', str(y), '-z', str(PIECE_Z)
            ],
            output='screen'
        )

    piece_spawns = []

    # rank indices: 0-based row -> chess rank (row 1 = white back rank, row 8 = black back rank)
    WHITE_BACK_ROW = 0
    WHITE_PAWN_ROW = 1
    BLACK_PAWN_ROW = 6
    BLACK_BACK_ROW = 7

    for col_idx in range(8):
        piece_spawns.append(make_piece_spawn('w', BACK_RANK_ORDER[col_idx], col_idx, WHITE_BACK_ROW))
        piece_spawns.append(make_piece_spawn('w', 'pawn', col_idx, WHITE_PAWN_ROW))
        piece_spawns.append(make_piece_spawn('b', 'pawn', col_idx, BLACK_PAWN_ROW))
        piece_spawns.append(make_piece_spawn('b', BACK_RANK_ORDER[col_idx], col_idx, BLACK_BACK_ROW))

    # Give the chessboard a moment to spawn first so pieces land on top of it, not mid-air/falling through
    delayed_piece_spawns = TimerAction(period=2.0, actions=piece_spawns)

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen'
    )

    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'panda', '-z', '0.0'],
        output='screen'
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )

    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['panda_arm_controller'],
        output='screen'
    )

    hand_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['panda_hand_controller'],
        output='screen'
    )

    # Spawn controllers only after the robot entity exists in Gazebo
    delayed_controllers = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[joint_state_broadcaster_spawner, arm_controller_spawner, hand_controller_spawner]
        )
    )

    return LaunchDescription([
        gz_resource_path,
        gazebo,
        table_spawn,
        chessboard_spawn,
        delayed_piece_spawns,
        camera_spawn,
        camera_bridge,
        camera_tf,
        clock_bridge,
        robot_state_publisher,
        spawn_entity,
        delayed_controllers,
        delayed_calibration
    ])