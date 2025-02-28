import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
import numpy as np

class RobotMover(Node):
    def __init__(self):
        super().__init__('robot_mover_node')

        self.subscription = self.create_subscription(
            Float32MultiArray,
            'robot_control',
            self.listener_callback,
            10)
            
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            10)
        
        self.status_pub = self.create_publisher(String, 'status_follow_me', 10)
            
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.threshold = 80.00
        
        self.detection_distance = 1.00
        
        self.front_angles = (-31, 31)    
        self.back_angles = (149, -149)  

    def status_publish(self, message):
        msg = String()
        msg.data = message
        self.status_pub.publish(msg)
        self.get_logger().info(message) 
        
    def get_region_distance(self, ranges, angle_range):
        start_idx = int((angle_range[0] + 180) * len(ranges) / 360)
        end_idx = int((angle_range[1] + 180) * len(ranges) / 360)

        if end_idx < start_idx:
            region_ranges = ranges[start_idx:] + ranges[:end_idx]
        else:
            region_ranges = ranges[start_idx:end_idx]
        valid_ranges = [r for r in region_ranges if r > 0.01 and not np.isinf(r)]
        
        return min(valid_ranges) if valid_ranges else float('inf')

    def lidar_callback(self, scan_msg):
        self.front_dist = self.get_region_distance(scan_msg.ranges, self.front_angles)
        self.back_dist = self.get_region_distance(scan_msg.ranges, self.back_angles)

    def listener_callback(self, msg):
        x_deviation, y_deviation = msg.data
        self.move_robot(x_deviation, y_deviation)

    def move_robot(self, x_deviation, y_deviation):
        max_linear_speed = 0.5
        max_angular_speed = 0.5
        twist = Twist()

        if x_deviation is not None and y_deviation is not None:
            # Check forward movement
            if y_deviation > 100:
                if self.front_dist < self.detection_distance:
                    self.status_publish("ตรวจเจอสิ่งกีดขวางด้านหน้า หยุดรอ!!!!")
                    twist.linear.x = 0.0
                else:
                    linear_speed = max_linear_speed * (y_deviation / (y_deviation + self.threshold))
                    twist.linear.x = min(linear_speed, max_linear_speed)
                    self.status_publish(f"เดินหน้าด้วยความเร็ว: {twist.linear.x:.2f}")
            
            # Check backward movement
            elif y_deviation < 50:
                if self.back_dist < self.detection_distance:
                    self.status_publish("ตรวจเจอสิ่งกีดขวางด้านหลัง หยุดรอ!!!!")
                    twist.linear.x = 0.0
                else:
                    linear_speed = -(max_linear_speed * ((50 - y_deviation) / ((50 - y_deviation) + self.threshold)))
                    twist.linear.x = min(linear_speed, max_linear_speed)
                    self.status_publish(f"ถอยหลังด้วยความเร็ว: {twist.linear.x:.2f}")
            else:
                twist.linear.x = 0.0
                self.status_publish("ระยะห่างเหมาะสม หยุดรอ")

            # Angular movement (rotation) remains unchanged
            if abs(x_deviation) > self.threshold:
                angular_speed = max_angular_speed * (x_deviation / (abs(x_deviation) + self.threshold))
                twist.angular.z = max(min(angular_speed, max_angular_speed), -max_angular_speed)
                if x_deviation > 0:
                    self.status_publish(f"หมุนซ้ายด้วยความเร็ว: {twist.angular.z:.2f}")
                else:
                    self.status_publish(f"หมุนขวาด้วยความเร็ว: {twist.angular.z:.2f}")
            else:
                twist.angular.z = 0.0
        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.status_publish("หยุดรอ จนกว่าจะถูกสั่งงาน")

        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    robot_mover = RobotMover()
    try:
        rclpy.spin(robot_mover)
    except KeyboardInterrupt:
        pass
    finally:
        robot_mover.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()