#!/usr/bin/env python3
"""
chess_logic_node

Tracks a chess game with python-chess. Two ways to drive it:

1. Manual: publish a UCI move on /chess_move (e.g. "e2e4").
2. Self-play: publish anything on /start_self_play and the node picks a
   random legal move for whichever color's turn it is, sends it to the
   action server, and — on success — automatically picks the next one.
   This makes the single arm play both White and Black.

Normal captures ARE now handled: the captured piece is moved to an
off-board holding square first, then the capturing piece moves to the
target square. Holding squares fill out LEFT-RIGHT (varying col) one
per capture, only stepping to a further-out row after 8 pieces of that
color have been captured — captured White pieces go to the left of the
board (col -1, -2, ...), captured Black pieces go to the right (col 8,
9, ...).

STILL NOT handled: castling (needs two coordinated piece moves), en
passant (captured piece isn't on the destination square), and
promotion (would need to swap the pawn model for a different piece).
These are detected and reported, not executed.

Run:
    ros2 run <your_package> chess_logic_node

Test (manual):
    ros2 topic pub --once /chess_move std_msgs/msg/String "data: 'e2e4'"

Test (self-play):
    ros2 topic pub --once /start_self_play std_msgs/msg/Empty {}

Dependency:
    pip install chess --break-system-packages
"""
import random

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String, Empty

import chess

from board_interfaces.action import PickPlacePiece

HOLDING_COLS_PER_ROW = 3  # tune to your arm's actual reach; keep this small

class ChessLogicNode(Node):
    def __init__(self):
        super().__init__('chess_logic_node')

        self.board = chess.Board()
        self.self_play_active = False
        self._action_client = ActionClient(self, PickPlacePiece, 'pick_place_piece')

        # Holding-square cycling counters — one per captured color.
        self.white_captured_count = 0  # captured White pieces -> col -1, -2, ... (left of board)
        self.black_captured_count = 0  # captured Black pieces -> col 8, 9, ... (right of board)

        self.create_subscription(String, 'chess_move', self.on_move_msg, 10)
        self.create_subscription(Empty, 'start_self_play', self.on_start_self_play, 10)
        self.create_subscription(Empty, 'stop_self_play', self.on_stop_self_play, 10)

        self.get_logger().info(
            'chess_logic_node ready. Send UCI moves on /chess_move, '
            'or publish to /start_self_play for the robot to play both sides.'
        )
        self.get_logger().info('\n' + str(self.board))

    # --- square <-> col/row conversion, matching the launch file's convention ---
    # file a-h -> col 0-7, rank 1-8 -> row 0-7 (chess.square_file/rank already do exactly this)
    def square_to_col_row(self, square: int):
        return chess.square_file(square), chess.square_rank(square)

    def next_holding_square(self, captured_color: chess.Color):
        if captured_color == chess.WHITE:
            count = self.white_captured_count
            self.white_captured_count += 1
            col = -1 - (count % HOLDING_COLS_PER_ROW)
        else:
            count = self.black_captured_count
            self.black_captured_count += 1
            col = 8 + (count % HOLDING_COLS_PER_ROW)
        row = count // HOLDING_COLS_PER_ROW
        return col, row

    # A move that can be executed as-is, with no capture/castle/promotion handling needed.
    def is_plain_move(self, move: chess.Move) -> bool:
        return not (
            self.board.is_castling(move)
            or self.board.is_en_passant(move)
            or self.board.is_capture(move)
            or move.promotion is not None
        )

    # A capture we know how to handle: captured piece sits right on the target square.
    def is_handleable_capture(self, move: chess.Move) -> bool:
        return (
            self.board.is_capture(move)
            and not self.board.is_en_passant(move)
            and not self.board.is_castling(move)
            and move.promotion is None
        )

    def is_executable(self, move: chess.Move) -> bool:
        return self.is_plain_move(move) or self.is_handleable_capture(move)

    # --- manual mode ---
    def on_move_msg(self, msg: String):
        move_str = msg.data.strip()
        try:
            move = chess.Move.from_uci(move_str)
        except ValueError:
            self.get_logger().error(f'"{move_str}" is not valid UCI notation.')
            return

        if move not in self.board.legal_moves:
            self.get_logger().error(f'"{move_str}" is not a legal move in the current position.')
            return

        if not self.is_executable(move):
            self.get_logger().warn(f'"{move_str}" needs castling/en-passant/promotion handling — not executed.')
            return

        self.execute_move(move)

    # --- self-play mode ---
    def on_start_self_play(self, _msg: Empty):
        if self.self_play_active:
            self.get_logger().warn('Self-play is already running.')
            return
        self.self_play_active = True
        self.get_logger().info('Self-play started.')
        self.play_next_self_move()

    def on_stop_self_play(self, _msg: Empty):
        self.self_play_active = False
        self.get_logger().info('Self-play stopped.')

    def play_next_self_move(self):
        if not self.self_play_active:
            return

        if self.board.is_game_over():
            self.get_logger().info(f'Game over: {self.board.result()}. Self-play stopped.')
            self.self_play_active = False
            return

        executable_moves = [m for m in self.board.legal_moves if self.is_executable(m)]
        if not executable_moves:
            self.get_logger().warn(
                'No legal move available that this node can execute (all remaining moves '
                'need castling/en-passant/promotion handling). Self-play stopped here.'
            )
            self.self_play_active = False
            return

        move = random.choice(executable_moves)
        self.execute_move(move)

    # --- move execution (handles plain moves and captures) ---
    def execute_move(self, move: chess.Move):
        move_str = move.uci()

        def finalize(ok: bool):
            if ok:
                self.board.push(move)
                self.get_logger().info(f'"{move_str}" complete.\n' + str(self.board))
                if self.self_play_active:
                    self.play_next_self_move()
            else:
                self.get_logger().error(f'"{move_str}" failed — board state NOT updated.')
                self.self_play_active = False

        if self.is_handleable_capture(move):
            captured_color = not self.board.turn  # opponent of the side moving
            hold_col, hold_row = self.next_holding_square(captured_color)
            cap_col, cap_row = self.square_to_col_row(move.to_square)

            self.get_logger().info(f'"{move_str}" is a capture — clearing target square first.')

            def after_clear(ok: bool):
                if not ok:
                    self.get_logger().error(f'"{move_str}" — failed to clear captured piece, aborting move.')
                    self.self_play_active = False
                    return
                src_col, src_row = self.square_to_col_row(move.from_square)
                tgt_col, tgt_row = self.square_to_col_row(move.to_square)
                self.send_pick_place(f'{move_str} (capturing move)', src_col, src_row, tgt_col, tgt_row, finalize)

            self.send_pick_place(
                f'{move_str} (clear capture)', cap_col, cap_row, hold_col, hold_row, after_clear
            )
        else:
            src_col, src_row = self.square_to_col_row(move.from_square)
            tgt_col, tgt_row = self.square_to_col_row(move.to_square)
            self.send_pick_place(move_str, src_col, src_row, tgt_col, tgt_row, finalize)

    # --- action-client plumbing ---
    def send_pick_place(self, label, source_col, source_row, target_col, target_row, done_cb):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('pick_place_piece action server not available.')
            done_cb(False)
            return

        goal = PickPlacePiece.Goal()
        goal.source_col = source_col
        goal.source_row = source_row
        goal.target_col = target_col
        goal.target_row = target_row

        self.get_logger().info(
            f'Sending "{label}" -> source=({source_col},{source_row}) target=({target_col},{target_row})'
        )
        future = self._action_client.send_goal_async(goal, feedback_callback=self.on_feedback)
        future.add_done_callback(lambda f: self.on_goal_response(f, label, done_cb))

    def on_feedback(self, feedback_msg):
        self.get_logger().info(f'Feedback: {feedback_msg.feedback.status}')

    def on_goal_response(self, future, label, done_cb):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f'Goal for "{label}" was rejected.')
            done_cb(False)
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda f: self.on_result(f, label, done_cb))

    def on_result(self, future, label, done_cb):
        result = future.result().result
        if not result.success:
            self.get_logger().error(f'"{label}" failed at the action server.')
        done_cb(result.success)


def main(args=None):
    rclpy.init(args=args)
    node = ChessLogicNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()