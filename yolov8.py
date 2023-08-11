from ultralytics import YOLO
import cv2

class YOLOv8:
    def __init__(self, model_name):
        self.model = YOLO(model_name)

    def detect(self, img):
        result = self.model.predict(source=img, conf=0.3, iou=0.3)[0]
        boxes = list(result.boxes.xyxy.cpu().data.numpy().astype(dtype=int))  # box with xyxy format, (N, 4)
        scores = list(result.boxes.conf.cpu().data.numpy())  # confidence score, (N, 1)
        classes = list(result.boxes.cls.cpu().data.numpy().astype(dtype=int))  # cls, (N, 1)

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
    detector = YOLOv8('yolov8m.pt')

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


