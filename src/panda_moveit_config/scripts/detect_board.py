import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2

import cv2
from cv_bridge import CvBridge
import numpy as np


class ChessPieceDetector(Node):
    def __init__(self):
        super().__init__('chess_piece_detector')

        self.pattern_size = (7, 7)  # Inner board corners

        self.files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.ranks = ['8', '7', '6', '5', '4', '3', '2', '1']

        self.bridge = CvBridge()
        self.latest_pc = None

        # Height threshold (meters): If depth decreases by > 1.5 cm, a piece is on the square
        self.height_threshold = 0.015

        self.create_subscription(
            PointCloud2,
            '/overhead_camera/points',
            self.pc_callback,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Image,
            '/overhead_camera/image',
            self.color_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info("Chess Piece Occupancy Detector Started.")

    def pc_callback(self, msg: PointCloud2):
        self.latest_pc = msg

    def color_callback(self, msg: Image):
        if self.latest_pc is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

        if not ret:
            return

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        cols, rows = self.pattern_size  # 7, 7
        inner_grid = corners.reshape((rows, cols, 2))

        # ------------------------------------------------------------------
        # STEP 1: Reconstruct 9x9 Grid (All 64 Squares) - in IMAGE pixel space
        # ------------------------------------------------------------------
        full_grid = np.zeros((9, 9, 2), dtype=np.float32)
        full_grid[1:8, 1:8] = inner_grid

        for r in range(1, 8):
            step_x_left = inner_grid[r - 1, 1] - inner_grid[r - 1, 0]
            full_grid[r, 0] = inner_grid[r - 1, 0] - step_x_left

            step_x_right = inner_grid[r - 1, 6] - inner_grid[r - 1, 5]
            full_grid[r, 8] = inner_grid[r - 1, 6] + step_x_right

        for c in range(0, 9):
            step_y_top = full_grid[2, c] - full_grid[1, c]
            full_grid[0, c] = full_grid[1, c] - step_y_top

            step_y_bottom = full_grid[7, c] - full_grid[6, c]
            full_grid[8, c] = full_grid[7, c] + step_y_bottom

        # Draw a rectangle around the full board (outer boundary of the 9x9 grid).
        # Perspective-corrected, so it's drawn as a 4-point polygon, not axis-aligned.
        board_outline = np.array([
            full_grid[0, 0],
            full_grid[0, 8],
            full_grid[8, 8],
            full_grid[8, 0],
        ], dtype=np.int32)
        cv2.polylines(frame, [board_outline], isClosed=True, color=(255, 0, 0), thickness=3)

        # ------------------------------------------------------------------
        # STEP 2: Read Depth Data & Detect Piece Occupancy
        # ------------------------------------------------------------------
        pc_array = point_cloud2.read_points_numpy(
            self.latest_pc, field_names=("x", "y", "z")
        )
        pc_width = self.latest_pc.width
        pc_height = self.latest_pc.height
        pc_grid = pc_array.reshape((pc_height, pc_width, 3))

        img_h, img_w = frame.shape[:2]

        # --- FIX: image and point cloud can be different resolutions.
        # Scale every pixel coordinate from image space -> point cloud space
        # before indexing pc_grid. If they're already equal, scale = 1.0.
        scale_x = pc_width / img_w
        scale_y = pc_height / img_h

        def to_pc_coords(u, v):
            pc_u = int(round(u * scale_x))
            pc_v = int(round(v * scale_y))
            pc_u = max(2, min(pc_u, pc_width - 3))
            pc_v = max(2, min(pc_v, pc_height - 3))
            return pc_u, pc_v

        board_occupancy = {}

        # Corner pixel coords -> point cloud coords for the board plane sample
        corner_u = np.round(inner_grid[:, :, 0] * scale_x).astype(int)
        corner_v = np.round(inner_grid[:, :, 1] * scale_y).astype(int)
        corner_u = np.clip(corner_u, 0, pc_width - 1)
        corner_v = np.clip(corner_v, 0, pc_height - 1)
        board_z_plane = np.nanmedian(pc_grid[corner_v, corner_u, 2])

        if np.isnan(board_z_plane):
            self.get_logger().warn("board_z_plane is NaN, skipping this frame.")
            return

        for r in range(8):
            for c in range(8):
                p1 = full_grid[r, c]
                p2 = full_grid[r, c + 1]
                p3 = full_grid[r + 1, c + 1]
                p4 = full_grid[r + 1, c]

                # Center in IMAGE pixel space (used for drawing + color ROI)
                center_u = int((p1[0] + p2[0] + p3[0] + p4[0]) / 4.0)
                center_v = int((p1[1] + p2[1] + p3[1] + p4[1]) / 4.0)
                center_u = max(2, min(center_u, img_w - 3))
                center_v = max(2, min(center_v, img_h - 3))

                # Same center, mapped into POINT CLOUD pixel space for depth lookup
                pc_u, pc_v = to_pc_coords(center_u, center_v)

                square_name = f"{self.files[c]}{self.ranks[r]}"

                # Sample 5x5 ROI region for depth to avoid noisy pixels
                roi_depth = pc_grid[pc_v - 2:pc_v + 3, pc_u - 2:pc_u + 3, 2]
                valid_depths = roi_depth[~np.isnan(roi_depth) & ~np.isinf(roi_depth)]

                if len(valid_depths) > 0:
                    min_square_z = np.min(valid_depths)  # Closest point (top of piece)

                    if (board_z_plane - min_square_z) > self.height_threshold:
                        # ----------------------------------------------
                        # STEP 3: Piece Color Classification (RGB ROI)
                        # ----------------------------------------------
                        roi_color = frame[center_v - 4:center_v + 5, center_u - 4:center_u + 5]
                        gray_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
                        avg_brightness = np.mean(gray_roi)

                        piece_color = "White Piece" if avg_brightness > 120 else "Black Piece"
                        board_occupancy[square_name] = piece_color

                        circle_color = (0, 255, 0) if piece_color == "White Piece" else (0, 0, 255)
                        cv2.circle(frame, (center_u, center_v), 10, circle_color, 2)
                    else:
                        board_occupancy[square_name] = "Empty"
                else:
                    board_occupancy[square_name] = "Unknown"

        # ------------------------------------------------------------------
        # STEP 4: Output Occupancy Map
        # ------------------------------------------------------------------
        occupied_only = {k: v for k, v in board_occupancy.items() if v != "Empty"}
        self.get_logger().info(f"Detected Occupied Squares: {occupied_only}")

        cv2.imshow("Detected Chess Pieces", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ChessPieceDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()