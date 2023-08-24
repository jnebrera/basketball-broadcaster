from yolov8 import YOLOv8
import norfair
import numpy as np
import cv2
from shapely import Polygon, LineString

def to_norfair_detections(boxes, scores, class_name):
    detections = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        data = {
            "name": class_name,
            "p": score,
        }
        box = np.array(
            [
                [x1, y1],
                [x2, y2],
            ]
        )
        detection = norfair.Detection(
            points=box,
            data=data,
        )
        detections.append(detection)
    return detections

class PBDetector:
    def __init__(self, court_region):
        self.detector = YOLOv8('yolov8n.pt') # 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'
        self.court_region = Polygon(court_region)

    def detect(self, img):
        boxes, scores, cls_ids = self.detector.detect(img)
        p_boxes = []
        p_scores = []
        b_boxes = []
        b_scores = []
        for box, score, cls_id in zip(boxes, scores, cls_ids):
            if cls_id == 0: # person
                feet_line = LineString([[box[0], box[3]], [box[2], box[3]]])
                if feet_line.within(self.court_region):
                    p_boxes.append(box)
                    p_scores.append(score)
            elif cls_id == 32: # sports ball
                b_boxes.append(box)
                b_scores.append(score)

        player_detections = to_norfair_detections(p_boxes, p_scores, 'player')
        ball_detections = to_norfair_detections(b_boxes, b_scores, 'ball')

        return player_detections, ball_detections

