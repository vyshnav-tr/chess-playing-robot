#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <board_interfaces/action/pick_place_piece.hpp>
#include <thread>
#include <chrono>
#include <memory>
#include <algorithm>
#include <cmath>
#include <iterator>

using PickPlacePiece = board_interfaces::action::PickPlacePiece;
using GoalHandlePickPlace = rclcpp_action::ServerGoalHandle<PickPlacePiece>;

class PickPlaceActionServer : public rclcpp::Node {
public:
    PickPlaceActionServer() : Node("pick_place_action_server",
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)) {}

    // Call this once after construction, since it needs shared_from_this()
    void initialize() {
        arm_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "panda_arm");
        hand_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "hand");
        arm_->setPoseReferenceFrame("panda_link0");
        arm_->setGoalOrientationTolerance(0.15);
        arm_->setMaxVelocityScalingFactor(0.9);
        arm_->setMaxAccelerationScalingFactor(0.9);
        arm_->setPlanningTime(10.0);         // default is too tight for constrained (height-boxed) plans
        arm_->setNumPlanningAttempts(15);    // more restarts with different seeds before giving up
        hand_->setMaxVelocityScalingFactor(0.9);
        hand_->setMaxAccelerationScalingFactor(0.9);

        planning_scene_interface_ = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();
        addCollisionObjects();

        // --- Board position is fixed/known (see BOARD_X/Y/Z, SQUARE_SIZE) — no calibration call needed ---
        resetToReady();

        // --- Start the action server, ready to accept move commands ---
        action_server_ = rclcpp_action::create_server<PickPlacePiece>(
            this, "pick_place_piece",
            std::bind(&PickPlaceActionServer::handleGoal, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&PickPlaceActionServer::handleCancel, this, std::placeholders::_1),
            std::bind(&PickPlaceActionServer::handleAccepted, this, std::placeholders::_1)
        );
        RCLCPP_INFO(get_logger(), "pick_place_piece action server ready.");
    }

private:
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_, hand_;
    std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;
    rclcpp_action::Server<PickPlacePiece>::SharedPtr action_server_;

    static constexpr double BOARD_X = 0.42, BOARD_Y = 0.0, BOARD_Z = 0.4;
    static constexpr double SQUARE_SIZE = 0.4 / 8.0;
    const double APPROACH_HEIGHT = 0.28;
    // Fixed grasp height for all piece types. Pawns are shorter than this reaches
    // by default — their collision box is stretched (invisible, not the visual mesh)
    // in fix_chess_pieces.py so they still make contact here. See that script's
    // PAWN_COLLISION_TOP comment if this value changes.
    static constexpr double GRASP_HEIGHT = 0.55;
    // True only until the first move completes. The very first "top of source" has
    // to free-plan from the arbitrary "ready" pose, but every move after that starts
    // from the previous move's "top of target (up)" — already at topHeight — so it
    // should travel there in a straight line instead of letting OMPL dip down.
    bool first_move_ = true;
    bool holding_piece_ = false;  // true between grasp-close and release-open

    geometry_msgs::msg::Pose makePose(double x, double y, double z) {
        geometry_msgs::msg::Pose pose;
        pose.position.x = x; pose.position.y = y; pose.position.z = z;
        pose.orientation.x = 1.0; pose.orientation.y = 0.0; pose.orientation.z = 0.0; pose.orientation.w = 0.0;
        return pose;
    }

    std::pair<double, double> squareCenter(int col, int row) {
        // col (a-h, the "file") runs left-right relative to the robot -> Y
        // row (the "rank"/side) runs toward/away from the robot -> X
        // Must match square_xy() in the spawn launch file, or targets won't line up with pieces.
        double x = BOARD_X + (row - 3.5) * SQUARE_SIZE;
        double y = BOARD_Y + (col - 3.5) * SQUARE_SIZE;
        return {x, y};
    }

    void addCollisionObjects() {
        std::vector<moveit_msgs::msg::CollisionObject> objects;

        moveit_msgs::msg::CollisionObject table;
        table.header.frame_id = "panda_link0";
        table.id = "table";
        shape_msgs::msg::SolidPrimitive table_shape;
        table_shape.type = table_shape.BOX;
        table_shape.dimensions = {0.6, 0.6, 0.4};
        geometry_msgs::msg::Pose table_pose;
        table_pose.position.x = BOARD_X; table_pose.position.y = BOARD_Y; table_pose.position.z = 0.2;
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
        board_pose.position.x = BOARD_X; board_pose.position.y = BOARD_Y; board_pose.position.z = 0.41;
        board_pose.orientation.w = 1.0;
        board.primitives.push_back(board_shape);
        board.primitive_poses.push_back(board_pose);
        board.operation = board.ADD;
        objects.push_back(board);

        planning_scene_interface_->applyCollisionObjects(objects);
    }

    void resetToReady() {
        arm_->setNamedTarget("ready");
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        if (arm_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            arm_->execute(plan);
        } else {
            RCLCPP_ERROR(get_logger(), "Failed to reach ready pose!");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Recovery "ready" used mid-move (e.g. from goToLevel's fallback) instead of the SRDF
    // named "ready" state. The named "ready" joint config has no guaranteed end-effector
    // orientation, so if a piece is in the gripper it could tip or slide during that move.
    // This instead uses a pose target with the SAME straight-down orientation as every
    // pick/place pose (makePose's fixed quaternion), parked safely above the board center —
    // gripper stays facing down the whole time, so a held piece stays seated.
    bool resetToReadyDown() {
        double safeZ = APPROACH_HEIGHT + BOARD_Z + 0.15;  // well above the board, clear of pieces
        return goTo(BOARD_X, BOARD_Y, safeZ, "ready (gripper down)");
    }

    bool goTo(double x, double y, double z, const char* label) {
        arm_->setPoseTarget(makePose(x, y, z));
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        if (arm_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            arm_->execute(plan);
            RCLCPP_INFO(get_logger(), "Reached: %s (%.3f, %.3f, %.3f)", label, x, y, z);
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
            return true;
        }
        RCLCPP_ERROR(get_logger(), "FAILED to reach: %s", label);
        return false;
    }

    // Straight-line Cartesian move — use for vertical descend/ascend so the
    // gripper goes straight down/up instead of swooping in at an angle.
    // jump_threshold is enabled (nonzero) so a sudden joint-space reconfiguration
    // can't happen mid-path — without it, the gripper stays on the line but other
    // links of the arm could swing wide and clip a neighboring piece.
    //
    // Retries a few times with progressively looser step-size/jump-threshold params
    // if the first attempt can't complete the line. This does NOT relax the line
    // itself — every attempt still requires a fully-straight (>99%) Cartesian path,
    // execution just fails or succeeds; it changes how hard the IK solver tries to
    // find a valid joint-space route along that same physical line.
    bool goToLinear(double x, double y, double z, const char* label) {
        struct Attempt { double eef_step; double jump_threshold; };
        static const Attempt attempts[] = {
            {0.005, 4.0},
            {0.01,  8.0},
            {0.02,  15.0},
        };

        geometry_msgs::msg::Pose target = makePose(x, y, z);
        std::vector<geometry_msgs::msg::Pose> waypoints = { target };

        for (size_t i = 0; i < std::size(attempts); ++i) {
            moveit_msgs::msg::RobotTrajectory trajectory;
            double fraction = arm_->computeCartesianPath(
                waypoints, attempts[i].eef_step, attempts[i].jump_threshold, trajectory);

            if (fraction > 0.99) {
                moveit::planning_interface::MoveGroupInterface::Plan plan;
                plan.trajectory_ = trajectory;
                arm_->execute(plan);
                RCLCPP_INFO(get_logger(), "Reached (linear): %s (%.3f, %.3f, %.3f) [attempt %zu/%zu]",
                    label, x, y, z, i + 1, std::size(attempts));
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                return true;
            }

            if (i + 1 < std::size(attempts)) {
                RCLCPP_WARN(get_logger(), "LINEAR attempt %zu/%zu failed (%.0f%% of path computed): %s — retrying with looser IK params, line unchanged.",
                    i + 1, std::size(attempts), fraction * 100, label);
            } else {
                RCLCPP_ERROR(get_logger(), "LINEAR FAILED after %zu attempts (%.0f%% of path computed): %s",
                    std::size(attempts), fraction * 100, label);
            }
        }
        return false;
    }

    // Used ONLY for horizontal "top of X" transit — moving between two points already
    // at (roughly) the same height. Doesn't require a strict single straight line like
    // goToLinear; it just must never dip DOWN toward the board — going higher is fine.
    // Lets OMPL free-plan (so it can route around singularities a locked Cartesian line
    // can't) while a path constraint enforces a hard floor with generous headroom above.
    //
    // IMPORTANT: the constraint box must cover BOTH the current EE height and the
    // target height z — not just z. MoveIt validates path constraints against the
    // start state too, so if the current pose sits below a floor fixed at z (e.g.
    // right after descending to GRASP_HEIGHT for a pick), the start state itself is
    // invalid and planning fails almost every attempt. We size the box from
    // min(currentZ, z) up through max(currentZ, z) + headroom instead.
    bool goToLevel(double x, double y, double z, const char* label, double headroom = 0.15) {
        double currentZ = arm_->getCurrentPose().pose.position.z;
        double currentX = arm_->getCurrentPose().pose.position.x;
        double currentY = arm_->getCurrentPose().pose.position.y;
        RCLCPP_INFO(get_logger(), "goToLevel(%s): current=(%.3f,%.3f,%.3f) -> target=(%.3f,%.3f,%.3f), headroom=%.2f",
            label, currentX, currentY, currentZ, x, y, z, headroom);
        double lowZ = std::min(z, currentZ) - 0.02;   // small buffer below whichever is lower
        double highZ = std::max(z, currentZ) + headroom;
        double boxHeight = highZ - lowZ;

        moveit_msgs::msg::PositionConstraint pos_constraint;
        pos_constraint.header.frame_id = "panda_link0";
        pos_constraint.link_name = arm_->getEndEffectorLink();
        pos_constraint.weight = 1.0;

        shape_msgs::msg::SolidPrimitive region;
        region.type = region.BOX;
        region.dimensions = {2.0, 2.0, boxHeight};  // wide in x/y; spans [lowZ, highZ] in z

        geometry_msgs::msg::Pose region_pose;
        region_pose.position.x = 0.5;              // box is wide enough that exact x/y doesn't matter
        region_pose.position.y = 0.0;
        region_pose.position.z = lowZ + boxHeight / 2.0;
        region_pose.orientation.w = 1.0;

        pos_constraint.constraint_region.primitives.push_back(region);
        pos_constraint.constraint_region.primitive_poses.push_back(region_pose);

        moveit_msgs::msg::Constraints constraints;
        constraints.position_constraints.push_back(pos_constraint);
        arm_->setPathConstraints(constraints);

        arm_->setPoseTarget(makePose(x, y, z));
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        bool ok = (arm_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
        if (ok) {
            arm_->execute(plan);
            RCLCPP_INFO(get_logger(), "Reached (level): %s (%.3f, %.3f, %.3f)", label, x, y, z);
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
        }
        arm_->clearPathConstraints();

        if (!ok) {
            if (holding_piece_) {
                RCLCPP_WARN(get_logger(), "LEVEL PLAN FAILED: %s (holding piece) — resetting to ready and retrying from there. Risk: piece may swing during this recovery.", label);
            } else {
                RCLCPP_WARN(get_logger(), "LEVEL PLAN FAILED: %s (1 try) — resetting to ready and retrying from there.", label);
            }
            bool readyOk = resetToReadyDown();
            if (!readyOk) {
                RCLCPP_WARN(get_logger(), "resetToReadyDown() failed for %s — falling back to SRDF 'ready' to at least get unstuck.", label);
                resetToReady();
            }
            ok = goTo(x, y, z, label);

            if (!ok && headroom < 0.6) {
                // Ready-reset didn't work either — fall back to widening the height band,
                // still never dropping below the floor (current/target height).
                RCLCPP_WARN(get_logger(), "Ready-reset retry also failed for %s — retrying with more headroom (%.2f -> %.2f).", label, headroom, headroom * 2.0);
                return goToLevel(x, y, z, label, headroom * 2.0);
            }
        }

        if (!ok) {
            RCLCPP_ERROR(get_logger(), "LEVEL PLAN FAILED: %s", label);
        }

        return ok;
    }

    bool moveGripper(double position, const char* label) {
        hand_->setJointValueTarget("panda_finger_joint1", position);
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        if (hand_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            hand_->execute(plan);
            RCLCPP_INFO(get_logger(), "Gripper: %s", label);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            return true;
        }
        RCLCPP_ERROR(get_logger(), "GRIPPER FAILED: %s", label);
        return false;
    }

    rclcpp_action::GoalResponse handleGoal(const rclcpp_action::GoalUUID &,
        std::shared_ptr<const PickPlacePiece::Goal> goal) {
        RCLCPP_INFO(get_logger(), "Goal received: (%d,%d) -> (%d,%d)",
            goal->source_col, goal->source_row, goal->target_col, goal->target_row);
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handleCancel(const std::shared_ptr<GoalHandlePickPlace>) {
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handleAccepted(const std::shared_ptr<GoalHandlePickPlace> goal_handle) {
        std::thread(std::bind(&PickPlaceActionServer::execute, this, goal_handle)).detach();
    }

    void execute(const std::shared_ptr<GoalHandlePickPlace> goal_handle) {
        auto goal = goal_handle->get_goal();
        auto feedback = std::make_shared<PickPlacePiece::Feedback>();
        auto result = std::make_shared<PickPlacePiece::Result>();

        auto [sx, sy] = squareCenter(goal->source_col, goal->source_row);
        auto [tx, ty] = squareCenter(goal->target_col, goal->target_row);
        double topHeight = APPROACH_HEIGHT + BOARD_Z;

        feedback->status = "Starting move";
        goal_handle->publish_feedback(feedback);

        moveGripper(0.025, "open");
        bool reachedSourceTop;
        if (first_move_) {
            reachedSourceTop = goTo(sx, sy, topHeight, "top of source");       // free plan (only from "ready")
            first_move_ = false;
        } else {
            reachedSourceTop = goToLevel(sx, sy, topHeight, "top of source");  // already near topHeight
        }
        if (!reachedSourceTop) {
            RCLCPP_ERROR(get_logger(), "Could not reach top of source at all — aborting this move.");
            result->success = false;
            goal_handle->succeed(result);
            return;
        }

        bool reachedGrasp = goToLinear(sx, sy, GRASP_HEIGHT, "source (down)");  // straight down

        if (reachedGrasp) {
            moveGripper(0.0, "close");
            holding_piece_ = true;
        }
        if (!goToLinear(sx, sy, topHeight, "top of source (up)")) {
            RCLCPP_ERROR(get_logger(), "Could not lift back to top of source — aborting this move (piece may be stranded mid-air/on source square).");
            result->success = false;
            goal_handle->succeed(result);
            return;
        }
        if (!goToLevel(tx, ty, topHeight, "top of target")) {
            RCLCPP_ERROR(get_logger(), "Could not reach top of target — aborting this move (piece is still in the gripper, not placed).");
            result->success = false;
            goal_handle->succeed(result);
            return;
        }
        bool reachedPlace = goToLinear(tx, ty, GRASP_HEIGHT, "target (down)");  // straight down

        if (reachedPlace) {
            moveGripper(0.020, "open (release)");
            holding_piece_ = false;
        }
        bool reachedFinalLift = goToLinear(tx, ty, topHeight, "top of target (up)");  // straight up — x/y unchanged
        if (!reachedFinalLift) {
            RCLCPP_ERROR(get_logger(), "Could not lift off target after placing — arm may be left near the board, next move's start pose could be affected.");
        }

        result->success = reachedGrasp && reachedPlace && reachedFinalLift;
        goal_handle->succeed(result);
        RCLCPP_INFO(get_logger(), "Move complete: success=%s", result->success ? "true" : "false");
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PickPlaceActionServer>();
    node->initialize();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}