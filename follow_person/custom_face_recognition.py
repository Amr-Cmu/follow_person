import cv2
import os
from tqdm import tqdm
import glob

class FaceRecognition:
    def __init__(self, data_directory):
        self.COSINE_THRESHOLD = 0.5
        self.MIN_FACE_SIZE = (30, 30)
        self.directory = data_directory
        
        # Initialize face detection model
        weights_face = os.path.join(self.directory, "/home/thanawat/amr_ws/src/follow_person//data/models/face_detection_yunet_2023mar.onnx")
        self.face_detector = cv2.FaceDetectorYN_create(weights_face, "", (0, 0))
        self.face_detector.setScoreThreshold(0.75)
        self.face_detector.setNMSThreshold(0.3)
        self.face_detector.setTopK(5)
        
        # Initialize face recognition model
        weights_recog = os.path.join(self.directory, "/home/thanawat/amr_ws/src/follow_person/data/models/face_recognizer_fast.onnx")
        self.face_recognizer = cv2.FaceRecognizerSF_create(weights_recog, "")
        
        # Load registered faces
        self.dictionary = self.load_registered_faces()
    
    def load_registered_faces(self):
        dictionary = {}
        types = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG')
        files = []
        for a_type in types:
            files.extend(glob.glob(os.path.join(self.directory, '/home/thanawat/amr_ws/src/follow_person/data/images', a_type)))
        
        files = list(set(files))
        print("Loading registered faces...")
        for file in tqdm(files):
            image = cv2.imread(file)
            if image is None:
                print(f"Warning: Could not load image {file}")
                continue
                
            image = self.preprocess_image(image)
            features, faces = self.recognize_face(image)
            
            if faces is None or len(faces) == 0:
                print(f"Warning: No face detected in {file}")
                continue
                
            user_id = os.path.splitext(os.path.basename(file))[0]
            dictionary[user_id] = features[0]
            
        print(f'Loaded {len(dictionary)} registered faces')
        return dictionary

    def preprocess_image(self, image):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
        max_dimension = 1000
        if max(image.shape[0], image.shape[1]) > max_dimension:
            scale = max_dimension / max(image.shape[0], image.shape[1])
            image = cv2.resize(image, None, fx=scale, fy=scale)
            
        return image

    def match(self, feature1):
        max_score = 0.0
        sim_user_id = ""
        for user_id, feature2 in self.dictionary.items():
            score = self.face_recognizer.match(
                feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
            if score >= max_score:
                max_score = score
                sim_user_id = user_id
        if max_score < self.COSINE_THRESHOLD:
            return False, ("", 0.0)
        return True, (sim_user_id, max_score)

    def recognize_face(self, image):
        height, width, _ = image.shape
        self.face_detector.setInputSize((width, height))
        
        try:
            _, faces = self.face_detector.detect(image)
            if faces is None:
                return None, None
                
            features = []
            valid_faces = []
            
            for face in faces:
                if face[2] >= self.MIN_FACE_SIZE[0] and face[3] >= self.MIN_FACE_SIZE[1]:
                    try:
                        aligned_face = self.face_recognizer.alignCrop(image, face)
                        feat = self.face_recognizer.feature(aligned_face)
                        features.append(feat)
                        valid_faces.append(face)
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue
                        
            return features, valid_faces
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return None, None