#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import base64

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        self.subscription = self.create_subscription(
            Image,
            'camera_feed',
            self.image_callback,
            10
        )
        
        self.base64_publisher = self.create_publisher(
            String,
            'camera/base64_image',
            10
        )
        
        self.bridge = CvBridge()
        
    def image_to_base64(self, cv_image):
        _, img_encoded = cv2.imencode('.jpg', cv_image)

        base64_string = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        
        return base64_string

    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        base64_string = self.image_to_base64(cv_image)
        string_msg = String()
        string_msg.data = base64_string
        self.base64_publisher.publish(string_msg)
        


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()