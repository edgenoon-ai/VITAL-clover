import time
import os
import numpy as np
import cv2

from engines.base_engine import BaseInferenceEngine


class YOLOXInferenceEngine(BaseInferenceEngine):
    def __init__(self, model_path: str, device: str, classes: int) -> None:
        super().__init__(model_path=model_path, device=device, classes=classes)

        self.target_size = (416, 416)

    def _preprocess(self, img: np.ndarray):
        conf = {}

        if len(img.shape) == 3:
            padded_img = np.full((*self.target_size, 3), 114, dtype=np.uint8)
        else:
            padded_img = np.full(*self.target_size, 114, dtype=np.uint8)

        r = min(self.target_size[0] / img.shape[0],
                self.target_size[1] / img.shape[1])

        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r),
                   : int(img.shape[1] * r)] = resized_img

        padded_img = np.expand_dims(padded_img.transpose((2, 0, 1)), 0)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

        conf['image'] = padded_img
        conf['shape'] = img.shape[:2]
        conf['ratio'] = r
        return conf

    def _postprocess(self, outputs, conf):
        outputs = outputs[self.outputs[0]].reshape(1, -1, 85)

        grids = []
        expanded_strides = []
        strides = [8, 16, 32]

        hsizes = [self.target_size[0] // stride for stride in strides]
        wsizes = [self.target_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        outputs = outputs[0]

        boxes = outputs[:, :4]
        scores = outputs[:, 4, None] * outputs[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= conf['ratio']
        dets = self.multiclass_nms(
            boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)

        if dets is not None:
            final_boxes = dets[:, :4]
            final_scores, final_cls_inds = dets[:, 4], dets[:, 5]

            return final_boxes.astype(np.int32), final_scores, final_cls_inds.astype(np.int32)

        return [], [], []

    @classmethod
    def nms(cls, boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    @classmethod
    def multiclass_nms(cls, boxes, scores, nms_thr, score_thr):
        # https://github.com/Megvii-BaseDetection/YOLOX/blob/dd5700c24693e1852b55ce0cb170342c19943d8b/yolox/utils/demo_utils.py#L80
        """Multiclass NMS implemented in Numpy. Class-agnostic version."""
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = cls.nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None],
                    valid_cls_inds[keep, None]], 1
            )
        return dets
