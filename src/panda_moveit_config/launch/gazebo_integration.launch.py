from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch, generate_moveit_rviz_launch
from launch import LaunchDescription

def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("panda", package_name="panda_moveit_config").to_moveit_configs()
    moveit_config.moveit_cpp.update({'use_sim_time': True})

    ld = LaunchDescription()
    ld.add_entity(generate_move_group_launch(moveit_config))
    ld.add_entity(generate_moveit_rviz_launch(moveit_config))
    return ld