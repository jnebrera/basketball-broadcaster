from . import footandball
import torch
import os
from . import augmentation
import cv2

class FAB:
    def __init__(self, ball_threshold=0.5, player_threshold=0.5):
        self.model = footandball.model_factory('fb1', 'detect', ball_threshold=ball_threshold, player_threshold=player_threshold)

        dir = os.path.dirname(__file__)
        model_path = os.path.join(dir, 'model_best_2023.09.10.pth')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(model_path, map_location=self.device)

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, img):
        img_tensor = augmentation.numpy2tensor(img)
        with torch.no_grad():
            # Add dimension for the batch size
            img_tensor = img_tensor.unsqueeze(dim=0).to(self.device)
            detections = self.model(img_tensor)[0]

        boxes = list(detections['boxes'].cpu().data.numpy().astype(dtype=int))  # box with xyxy format, (N, 4)
        scores = list(detections['scores'].cpu().data.numpy())  # confidence score, (N, 1)
        classes = list(detections['labels'].cpu().data.numpy().astype(dtype=int))  # cls, (N, 1)

        return boxes, scores, classes

    def draw_detection(self, image, boxes, scores, classes):
        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        for box, score, class_id in zip(boxes, scores, classes):
            color = (0, 255, 0)

            x1, y1, x2, y2 = box.astype(int)

            # Draw rectangle
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

            caption = '%.2f' % (score)
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                          fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)

            cv2.rectangle(det_img, (x1, y1),
                          (x1 + tw, y1 - th), color, -1)
            cv2.putText(det_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        return det_img


if __name__ == '__main__':
    detector = FAB()

    cap = cv2.VideoCapture('videos/test1_clip.mp4')

    while True:
        ok, image = cap.read()
        if not ok:
            break

        boxes, scores, classes = detector.detect(image)
        result_image = detector.draw_detection(image, boxes, scores, classes)

        result_image = cv2.resize(result_image, None, fx=0.5, fy=0.5)
        cv2.imshow('output', result_image)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()