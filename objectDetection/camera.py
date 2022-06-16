import sys
import time
import cv2
import numpy as np
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

Hitung = 0

class VideoCamera(object):
    def __init__ (self):
        self.cap = cv2.VideoCapture(0)
        # Deklarasi Variabel FPS
        self.counter, self.fps =0, 0
        self.start_time = time.time()
        self.fps_avg_frame_count = 10

        # Inialisasi Object Detection
        base_options = core.BaseOptions(file_name="efficientdet_lite2.tflite", use_coral=False, num_threads=4)
        detection_options = processor.DetectionOptions(max_results=10, score_threshold=0.3)
        options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
        self.detector = vision.ObjectDetector.create_from_options(options)

    def __del__ (self):
        self.cap.release()
     
    def get_frame(self):
        sucess, image = self.cap.read()
        self.counter += 1
        #image = cv2.flip(image, 0)
        
        # Konversi BGR Ke RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Membuat Tensor Gambar Objek Yang diDeteksi Dari rgb_image 
        input_tensor = vision.TensorImage.create_from_array(rgb_image)
        
        # Menjalankan Object Detection 
        detection_result = self.detector.detect(input_tensor)

        image = visualize(image, detection_result)

        if self.counter % self.fps_avg_frame_count == 0:
            end_time = time.time()
            self.fps = self.fps_avg_frame_count / (end_time - self.start_time)
            self.start_time = time.time()

        cv2.putText(image,'FPS = {:.1f}'.format(self.fps),(20,25),cv2.FONT_HERSHEY_PLAIN,2,(255,200,200),3)
        cv2.putText(image, 'Jumlah Barang:' + str(Hitung), (20,50), cv2.FONT_HERSHEY_PLAIN, 2 , (255,200,200), 3)
        sucess, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

def visualize(image: np.ndarray, detection_result: processor.DetectionResult) -> np.ndarray:
    global Hitung
    Hitung = 0
    for detection in detection_result.detections:
        category = detection.classes[0]
        probability = round(category.score, 2)
        class_name = category.class_name

        if  class_name != 'person' and class_name != 'dog' and class_name != 'cat':
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, (0, 0, 255), 3)

            # Draw label and score
            #result_text = class_name + ' (' + str(probability) + ')'
            result_text = class_name 
            text_location = (20 + bbox.origin_x, 20 + 20 + bbox.origin_y)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, 2 , (0, 0, 255), 3)

            Hitung = Hitung + 1

    return image