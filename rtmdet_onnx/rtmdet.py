# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import cv2
import onnxruntime

class RTMDet:
    def __init__(self, path, score_threshold=0.5, nms_threshold=0.95):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        self.session = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider',
                                                                     'CPUExecutionProvider'])
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def detect(self, org_img):
        img_h, img_w = org_img.shape[:2]
        input_w = 800
        input_h = input_w / img_w * img_h

        input_h = int(np.ceil(input_h / 32) * 32)
        input_w = int(np.ceil(input_w / 32) * 32)

        img = cv2.resize(org_img, (input_w, input_h))
        img = img.astype(np.float32)
        mean = [103.53, 116.28, 123.675]
        std = [57.375, 57.12, 58.395]
        img = img - mean
        img = img / std

        img = img.transpose(2, 0, 1)  # convert HWC to CHW
        img = img[np.newaxis, :, :, :]  # add batch dimension
        input_tensor = img.astype(np.float32)

        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        boxes = outputs[0][0]
        labels = outputs[1][0]
        scores = boxes[..., 4]
        boxes = boxes[..., :4]

        idx = cv2.dnn.NMSBoxes(boxes, scores, self.score_threshold, self.nms_threshold)

        if len(idx) > 0:
            boxes = boxes[idx]
            scores = scores[idx]
            labels = labels[idx]

            input_shape = np.array([input_w, input_h, input_w, input_h])
            boxes = np.divide(boxes, input_shape, dtype=np.float32)
            boxes *= np.array([img_w, img_h, img_w, img_h])
        else:
            boxes = np.empty((0, 4))
            scores = np.empty((0, 1))
            labels = np.empty((0, 1))

        return boxes.astype(int), scores, labels

if __name__ == '__main__':
    import time
    detector = RTMDet('../model/end2end.onnx')
    img = cv2.imread('../test_data/demo-0.jpg')

    for i in range(5):
        st = time.time()
        boxes, scores, classes = detector.detect(img)
        elapsed = time.time() - st
        print('Elapsed: {}s'.format(elapsed))

    ball_center_points = []
    cue_ball_index = -1

    for idx, (box, cls) in enumerate(zip(boxes, classes)):
        cx = (box[2] + box[0]) // 2
        cy = (box[3] + box[1]) // 2
        ball_center_points.append([int(cx), int(cy)])
        ball_color = (0, 255, 0)
        if cls == 1:
            cue_ball_index = idx
            ball_color = (0, 0, 255)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), ball_color, 2)

    cv2.imshow('output', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
