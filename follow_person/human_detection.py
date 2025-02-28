import cv2
import os

class HumanDetection:
    def __init__(self, data_directory):
        self.directory = data_directory
        # Initialize human detection model
        self.configPath = os.path.join(self.directory, '/home/thanawat/amr_ws/src/follow_person/data/models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
        self.weightsPath = os.path.join(self.directory,'/home/thanawat/amr_ws/src/follow_person/data/models/frozen_inference_graph.pb')
        self.human_detector = cv2.dnn_DetectionModel(self.weightsPath, self.configPath)
        self.human_detector.setInputSize(320, 320)
        self.human_detector.setInputScale(1.0/127.5)
        self.human_detector.setInputMean((127.5, 127.5, 127.5))
        self.human_detector.setInputSwapRB(True)

    def detect_humans(self, frame, conf_threshold=0.5, nms_threshold=0.3):
        return self.human_detector.detect(frame, confThreshold=conf_threshold, nmsThreshold=nms_threshold)