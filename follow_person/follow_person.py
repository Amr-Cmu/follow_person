import rclpy
from rclpy.node import Node
import cv2
import time
import sys
import cvzone
import numpy as np
sys.path.append('/home/thanawat/amr_ws/src/follow_person/follow_person')
from custom_face_recognition import FaceRecognition
from human_detection import HumanDetection
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

class PersonTracker:
    def __init__(self, name, bbox):
        self.name = name
        self.bbox = bbox
        self.is_tracked = True

class ImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        
    def process_frame(self, frame):
        resized_frame = cv2.resize(frame, (240, 160))
        ros_image = self.bridge.cv2_to_imgmsg(resized_frame, encoding="bgr8")
        return ros_image

class RobotControl:
    def __init__(self):
        self.robot_state = "running"  
    
    def process_frame(self, frame, boxes, labels, robot_state):
        if robot_state == "pause":
            cv2.putText(frame, 
                       "Paused",
                       (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6,
                       (0, 165, 255),  
                       2)
            return None

        _, width = frame.shape[:2]
        
        person_boxes = [box for box, label in zip(boxes, labels) if "Person" in label]
        tracking_boxes = [box for box, label in zip(boxes, labels) if "Tracking" in label]
        
        if not (len(person_boxes) == 1 and len(tracking_boxes) == 0) and \
           not (len(tracking_boxes) == 1 and len(person_boxes) == 0):
            cv2.putText(frame, 
                       "Not Sending",
                       (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6,
                       (0, 0, 255),  
                       2)
            return None
            
        if (len(person_boxes) == 1 and len(tracking_boxes) == 0):
            valid_boxes = person_boxes
        else:
            valid_boxes = tracking_boxes
            
        x, y, w, _ = valid_boxes[0]
        center_x_obj = x + w // 2
        x_deviation = round((width // 2) - center_x_obj, 3)
        y_deviation = round(y, 3)
        
        cv2.putText(frame, 
                   "Sending",
                   (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6,
                   (0, 255, 0), 
                   2)
        
        return x_deviation, y_deviation
        
class CombinedSystem:
    def __init__(self, data_directory, person_name):
        self.face_recognition = FaceRecognition(data_directory)
        self.human_detection = HumanDetection(data_directory)
        self.robot_control = RobotControl()  
        self.tracked_person = None
        self.target_name = person_name
        self.fixed_box_width = 200
        
    def update_tracker(self, tracker, bbox):
        tracker.bbox = bbox
        tracker.is_tracked = True

    def get_fixed_width_box(self, face_center_x, center_y, h):
        x = face_center_x - self.fixed_box_width // 2
        y = center_y - h // 2
        return [x, y, self.fixed_box_width, h]
    
    def process_frame(self, frame, robot_state):
        frame = self.face_recognition.preprocess_image(frame)
        
        detected_boxes = []
        detected_labels = []
        bbox_created = False
        
        classIds, confs, bbox = self.human_detection.detect_humans(frame)
        
        if len(classIds) > 0:
            for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
                if classId == 1 and conf > 0.50:
                    if bbox_created:
                        continue
                        
                    x, y, w, h = box
                    margin = 40
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    w = min(frame.shape[1] - x, w + 2*margin)
                    h = min(frame.shape[0] - y, h + 2*margin)
                    
                    person_region = frame[y:y+h, x:x+w]
                    if person_region.size == 0:
                        continue
                    
                    features, faces = self.face_recognition.recognize_face(person_region)
                    target_detected = False
                    
                    if faces is not None and len(faces) > 0:
                        for idx, (face, feature) in enumerate(zip(faces, features)):
                            result, user = self.face_recognition.match(feature)
                            
                            if result:
                                id_name, score = user
                                
                                if id_name == self.target_name:
                                    target_detected = True
                                    
                                    face_box = list(map(int, face[:4]))
                                    face_box_global_x = face_box[0] + x
                                    face_box_global_y = face_box[1] + y
                                    
                                    face_center_x = face_box_global_x + face_box[2] // 2
                                    face_center_y = face_box_global_y + face_box[3] // 2
                                    
                                    center_x = box[0] + box[2] // 2
                                    center_y = box[1] + box[3] // 2
                                    
                                    distance = np.sqrt((face_center_x - center_x)**2 + (face_center_y - center_y)**2)
                                    
                                    if distance < box[3] * 0.5:
                                        if self.tracked_person is None:
                                            self.tracked_person = PersonTracker(self.target_name, box)
                                        
                                        self.update_tracker(self.tracked_person, box)
                                        
                                        cv2.rectangle(frame, 
                                                    (face_box_global_x, face_box_global_y),
                                                    (face_box_global_x + face_box[2], face_box_global_y + face_box[3]),
                                                    (0, 255, 0), 2)
                                        
                                        text = f"{id_name} ({score:.2f})"
                                        cv2.putText(frame, text,
                                                (face_box_global_x, face_box_global_y - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                        
                                        fixed_box = self.get_fixed_width_box(face_center_x, center_y, box[3])
                                        cvzone.cornerRect(frame, fixed_box, colorC=(0, 255, 0))
                                        cv2.circle(frame, (face_center_x, center_y), 5, (255, 0, 0), -1)
                                        cv2.putText(frame, "Person",
                                                (fixed_box[0], fixed_box[1] - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                        
                                        detected_boxes.append(fixed_box)
                                        detected_labels.append("Person")
                                        bbox_created = True
                                    
                    if self.tracked_person and not target_detected and not bbox_created:
                        tracked_box = self.tracked_person.bbox
                        tracked_center = (tracked_box[0] + tracked_box[2]//2, tracked_box[1] + tracked_box[3]//2)
                        curr_center = (box[0] + box[2]//2, box[1] + box[3]//2)
                        
                        distance = np.sqrt((tracked_center[0] - curr_center[0])**2 + (tracked_center[1] - curr_center[1])**2)
                        
                        if distance < box[3] * 0.4:
                            center_x = box[0] + box[2] // 2
                            fixed_box = self.get_fixed_width_box(center_x, tracked_center[1], box[3])
                            
                            self.update_tracker(self.tracked_person, fixed_box)
                            cvzone.cornerRect(frame, fixed_box, colorC=(0, 165, 255))
                            cv2.circle(frame, (center_x, tracked_center[1]), 5, (255, 0, 0), -1)
                            cv2.putText(frame, f"Tracking {self.tracked_person.name}",
                                    (fixed_box[0], fixed_box[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                            
                            detected_boxes.append(fixed_box)
                            detected_labels.append("Tracking")
                            bbox_created = True
        
        control_values = self.robot_control.process_frame(frame, detected_boxes, detected_labels, robot_state)
        
        return frame, control_values

    def __del__(self):
        cv2.destroyAllWindows()

class FaceRecognitionNode(Node):
    def __init__(self):
        super().__init__('face_recognition_node')
        
        self.control_publisher = self.create_publisher(
            Float32MultiArray,
            'robot_control',
            10
        )
        
        self.image_publisher = self.create_publisher(
            Image,
            'camera_feed',
            10
        )

        self.pause_subscriber = self.create_subscription(String, '/pause', self.pause_callback, 10)
        self.robot_state = "running"

        self.declare_parameter('person_name', 'Default_Name')
        person_name = self.get_parameter('person_name').value

        self.data_directory = '/home/thanawat/amr_ws/src/follow_person/data' 
        self.system = CombinedSystem(self.data_directory, person_name)
        self.image_processor = ImageProcessor()
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.get_logger().error("Error: Could not open camera")
            sys.exit()
        
        self.timer = self.create_timer(0.033, self.process_frame) 
        self.last_time = time.time()
        self.get_logger().info('Face Recognition Node has been started')

    def pause_callback(self, msg):
        if msg.data == 'pause':
            self.robot_state = 'pause'
        else:
            self.robot_state = 'running'
            

    def process_frame(self):
        try:
            ret, current_frame = self.capture.read()
            if not ret:
                self.get_logger().error("Error: Could not read frame")
                return
            
            processed_frame, control_values = self.system.process_frame(current_frame, self.robot_state)
            
            if control_values is not None:
                msg = Float32MultiArray()
                msg.data = [float(control_values[0]), float(control_values[1])]
                self.control_publisher.publish(msg)
            
            current_time = time.time()
            fps = 1 / (current_time - self.last_time)
            self.last_time = current_time
            
            self.margin = 80
            height, width, _ = processed_frame.shape
            center_x = width // 2
            cv2.line(processed_frame, (center_x - self.margin, 0), 
                     (center_x - self.margin, height), (0, 0, 255), 2)
            cv2.line(processed_frame, (center_x + self.margin, 0), 
                     (center_x + self.margin, height), (0, 0, 255), 2)
            cv2.line(processed_frame, (center_x - 100, 100), (center_x + 100, 100), (0, 255, 255), 2)
            cv2.line(processed_frame, (center_x - 100, 50), (center_x + 100, 50), (0, 255, 255), 2)
            
            cv2.putText(processed_frame, f'FPS: {int(fps)}', (10, 20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            
            ros_image = self.image_processor.process_frame(processed_frame)
            self.image_publisher.publish(ros_image)
            cv2.imshow("Face Recognition System", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.destroy_node()
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {str(e)}')

    def __del__(self):
        self.capture.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognitionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.capture.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
