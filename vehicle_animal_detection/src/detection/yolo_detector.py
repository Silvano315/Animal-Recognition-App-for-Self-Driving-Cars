import cv2
import numpy as np

class YOLOTinyDetector:
    def __init__(self, config):
        self.config = config
        self.net = cv2.dnn.readNet(
            self.config['models']['yolo_tiny']['weights'],
            self.config['models']['yolo_tiny']['config']
        )
        with open(self.config['models']['yolo_tiny']['classes'], 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        self.conf_threshold = self.config['models']['yolo_tiny']['confidence_threshold']
        self.nms_threshold = self.config['models']['yolo_tiny']['nms_threshold']

        self.animal_classes = ['dog', 'cat', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']


    def detect(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype('int')
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        animal_detections = []
        for i in indices:
            i = i[0] if isinstance(i, tuple) else i
            box = boxes[i]
            x, y, w, h = box
            class_name = self.classes[class_ids[i]]
            if class_name in self.animal_classes:
                animal_detections.append({
                    'bbox': [x, y, x+w, y+h],
                    'class': class_name,
                    'confidence': confidences[i]
                })

        return animal_detections