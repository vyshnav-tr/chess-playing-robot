#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <thread>
#include <chrono>

const double BOARD_X = 0.5, BOARD_Y = 0.0, BOARD_Z = 0.4;
const double SQUARE_SIZE = 0.4 / 8.0;
const double APPROACH_HEIGHT = 0.25;   // top height above the board — must be clearly higher than GRASP_HEIGHT
const double GRASP_HEIGHT = 0.55;      // height at which the gripper actually reaches the piece

std::pair<double, double> squareCenter(int col, int row) {
    double x = BOARD_X + (col - 3.5) * SQUARE_SIZE;
    double y = BOARD_Y + (row - 3.5) * SQUARE_SIZE;
    return {x, y};
}

geometry_msgs::msg::Pose makePose(double x, double y, double z) {
    geometry_msgs::msg::Pose pose;
    pose.position.x = x;
    pose.position.y = y;
    pose.position.z = z;
    pose.orientation.x = 1.0;
    pose.orientation.y = 0.0;
    pose.orientation.z = 0.0;
    pose.orientation.w = 0.0;
    return pose;
}

void addCollisionObjects(moveit::planning_interface::PlanningSceneInterface &scene) {
    std::vector<moveit_msgs::msg::CollisionObject> objects;

    moveit_msgs::msg::CollisionObject table;
    table.header.frame_id = "panda_link0";
    table.id = "table";
    shape_msgs::msg::SolidPrimitive table_shape;
    table_shape.type = table_shape.BOX;
    table_shape.dimensions = {0.6, 0.6, 0.4};
    geometry_msgs::msg::Pose table_pose;
    table_pose.position.x = 0.5;
    table_pose.position.y = 0.0;
    table_pose.position.z = 0.2;
    table_pose.orientation.w = 1.0;
    table.primitives.push_back(table_shape);
    table.primitive_poses.push_back(table_pose);
    table.operation = table.ADD;
    objects.push_back(table);

    moveit_msgs::msg::CollisionObject board;
    board.header.frame_id = "panda_link0";
    board.id = "chessboard";
    shape_msgs::msg::SolidPrimitive board_shape;
    board_shape.type = board_shape.BOX;
    board_shape.dimensions = {0.4, 0.4, 0.02};
    geometry_msgs::msg::Pose board_pose;
    board_pose.position.x = 0.5;
    board_pose.position.y = 0.0;
    board_pose.position.z = 0.41;
    board_pose.orientation.w = 1.0;
    board.primitives.push_back(board_shape);
    board.primitive_poses.push_back(board_pose);
    board.operation = board.ADD;
    objects.push_back(board);

    scene.applyCollisionObjects(objects);
}

// One target, one plan, one execute. Nothing fancy — exactly one waypoint at a time.
bool goTo(moveit::planning_interface::MoveGroupInterface &arm, double x, double y, double z, const char* label) {
    arm.setPoseTarget(makePose(x, y, z));
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    if (arm.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
        arm.execute(plan);
        RCLCPP_INFO(rclcpp::get_logger("pick_place"), "Reached: %s (%.3f, %.3f, %.3f)", label, x, y, z);
        std::this_thread::sleep_for(std::chrono::seconds(3));
        return true;
    } else {
        RCLCPP_ERROR(rclcpp::get_logger("pick_place"), "FAILED to reach: %s (%.3f, %.3f, %.3f)", label, x, y, z);
        return false;
    }
}


bool moveGripper(moveit::planning_interface::MoveGroupInterface &hand, double position, const char* label) {
    hand.setJointValueTarget("panda_finger_joint1", position);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    if (hand.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
        hand.execute(plan);
        RCLCPP_INFO(rclcpp::get_logger("pick_place"), "Gripper: %s", label);
        std::this_thread::sleep_for(std::chrono::seconds(3));
        return true;
    } else {
        RCLCPP_ERROR(rclcpp::get_logger("pick_place"), "GRIPPER FAILED: %s", label);
        return false;
    }
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("pick_place_chess",
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

    moveit::planning_interface::MoveGroupInterface arm(node, "panda_arm");
    moveit::planning_interface::MoveGroupInterface hand(node, "hand");
    arm.setPoseReferenceFrame("panda_link0");

    // Step 0: ready
    arm.setNamedTarget("ready");
    moveit::planning_interface::MoveGroupInterface::Plan ready_plan;
    if (arm.plan(ready_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
        arm.execute(ready_plan);
        RCLCPP_INFO(node->get_logger(), "Reached: ready pose");
    } else {
        RCLCPP_ERROR(node->get_logger(), "FAILED to reach ready pose!");
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));

    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
    addCollisionObjects(planning_scene_interface);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    auto [sx, sy] = squareCenter(0, 0);
    auto [tx, ty] = squareCenter(4, 4);
    double topHeight = APPROACH_HEIGHT + BOARD_Z;

    RCLCPP_INFO(node->get_logger(), "Source: (%.3f, %.3f)  Target: (%.3f, %.3f)", sx, sy, tx, ty);

    // Ordered waypoint list, exactly like a Nav2 waypoint queue — each one its own
    // independent goal: plan -> execute -> wait 3s -> next waypoint.
    moveGripper(hand, 0.025, "open");
    goTo(arm, sx, sy, topHeight, "WAYPOINT 1: top of source");
    goTo(arm, sx, sy, GRASP_HEIGHT, "WAYPOINT 2: source (down)");
    moveGripper(hand, 0.0, "close");
    goTo(arm, sx, sy, topHeight, "WAYPOINT 3: top of source (up)");
    goTo(arm, tx, ty, topHeight, "WAYPOINT 4: top of target");
    goTo(arm, tx, ty, GRASP_HEIGHT, "WAYPOINT 5: target (down)");
    moveGripper(hand, 0.020, "open (release)");
    goTo(arm, tx, ty, topHeight, "WAYPOINT 6: top of target (up)");

    RCLCPP_INFO(node->get_logger(), "Pick and place complete.");
    rclcpp::shutdown();
    return 0;
}