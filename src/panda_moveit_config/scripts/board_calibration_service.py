#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped
from board_interfaces.srv import GetBoardPose

class BoardCalibrationService(Node):
    def __init__(self):
        super().__init__('board_calibration_service')
        self.bridge = CvBridge()
        self.depth_image = None
        self.camera_info = None
        self.latest_rgb = None
        self.depth_frame_id = None   # <-- capture the REAL frame_id Gazebo actually publishes

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, '/overhead_camera/image', self.rgb_cb, 10)
        self.create_subscription(Image, '/overhead_camera/depth_image', self.depth_cb, 10)
        self.create_subscription(CameraInfo, '/overhead_camera/camera_info', self.info_cb, 10)

        self.srv = self.create_service(GetBoardPose, 'get_board_pose', self.handle_request)
        self.get_logger().info('Board calibration service ready.')

    def rgb_cb(self, msg):
        self.latest_rgb = msg

    def depth_cb(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        if self.depth_frame_id is None:
            self.depth_frame_id = msg.header.frame_id
            self.get_logger().info(f'DEBUG: depth image real frame_id = "{self.depth_frame_id}"')

    def info_cb(self, msg):
        self.camera_info = msg

    def pixel_to_camera_point(self, u, v):
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]
        depth = self.depth_image[int(v), int(u)]
        if depth <= 0 or np.isnan(depth):
            return None
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        return x, y, z

    def transform_to_base(self, x, y, z, source_frame):
        pt = PointStamped()
        pt.header.frame_id = source_frame
        pt.header.stamp = rclpy.time.Time().to_msg()
        pt.point.x, pt.point.y, pt.point.z = float(x), float(y), float(z)
        try:
            transform = self.tf_buffer.lookup_transform('panda_link0', source_frame, rclpy.time.Time())
            t = transform.transform.translation
            r = transform.transform.rotation
            self.get_logger().info(f'DEBUG: transform panda_link0 <- {source_frame}: '
                                    f'trans=({t.x:.3f},{t.y:.3f},{t.z:.3f}) '
                                    f'rot=({r.x:.3f},{r.y:.3f},{r.z:.3f},{r.w:.3f})')
            transformed = do_transform_point(pt, transform)
            return transformed.point.x, transformed.point.y, transformed.point.z
        except Exception as e:
            self.get_logger().error(f'TF transform failed (source_frame="{source_frame}"): {e}')
            return None

    def handle_request(self, request, response):
        if self.latest_rgb is None or self.depth_image is None or self.camera_info is None:
            self.get_logger().error('DEBUG: missing rgb/depth/camera_info — not ready yet')
            response.success = False
            return response

        frame = self.bridge.imgmsg_to_cv2(self.latest_rgb, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (7, 7))
        if found:
            debug_frame = frame.copy()
            cv2.drawChessboardCorners(debug_frame, (7, 7), corners, found)
            u0, v0 = corners[0][0]
            cv2.circle(debug_frame, (int(u0), int(v0)), 10, (0, 0, 255), 3)
            cv2.putText(debug_frame, "CORNER[0]", (int(u0)+15, int(v0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite('/tmp/board_debug.png', debug_frame)
            
        if not found:
            self.get_logger().error('Board not detected')
            response.success = False
            return response

        u, v = corners[0][0]
        self.get_logger().info(f'DEBUG: raw pixel u={u:.1f}, v={v:.1f}')
        cam_point = self.pixel_to_camera_point(u, v)
        self.get_logger().info(f'DEBUG: camera-frame point (before transform): {cam_point}')
        if cam_point is None:
            response.success = False
            return response

        # Use the REAL frame_id from the depth image itself, not a hardcoded guess
        source_frame = self.depth_frame_id if self.depth_frame_id else 'overhead_camera_frame'
        base_point = self.transform_to_base(*cam_point, source_frame)
        self.get_logger().info(f'DEBUG: base-frame point (after transform): {base_point}')
        if base_point is None:
            response.success = False
            return response

        u2, v2 = corners[1][0]
        cam_point2 = self.pixel_to_camera_point(u2, v2)
        square_size = 0.05
        if cam_point2 is not None:
            dx = cam_point2[0] - cam_point[0]
            dy = cam_point2[1] - cam_point[1]
            square_size = float(np.sqrt(dx**2 + dy**2))

        response.origin_x = base_point[0]
        response.origin_y = base_point[1]
        response.origin_z = base_point[2]
        response.square_size = square_size
        response.success = True
        self.get_logger().info(f'Board origin in base frame: ({base_point[0]:.3f}, {base_point[1]:.3f}, {base_point[2]:.3f}), square_size={square_size:.3f}')
        return response

def main():
    rclpy.init()
    rclpy.spin(BoardCalibrationService())

if __name__ == '__main__':
    main()