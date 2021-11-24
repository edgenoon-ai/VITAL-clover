import time
import os
import numpy as np

from openvino.inference_engine import IECore


class BaseInferenceEngine:
    def __init__(self, model_path: str, device: str, classes: int) -> None:
        ie = IECore()

        net = ie.read_network(model_path, os.path.splitext(model_path)[0] + ".bin")
        self.input_blob = list(iter(net.input_info))[0]
        self.outputs = list(iter(net.outputs))

        if net.input_info[self.input_blob].input_data.shape[1] == 3:
            self.input_height, self.input_width = net.input_info[self.input_blob].input_data.shape[2:]
            self.nchw_shape = True
        else:
            self.input_height, self.input_width = net.input_info[self.input_blob].input_data.shape[1:3]
            self.nchw_shape = False
        
        self.engine = ie.load_network(network=net, device_name=device)
        self.classes = classes

        # net.input_info[self.input_blob].precision = 'FP32'
        # net.outputs[self.outputs[0]].precision = 'FP16'

    def __call__(self, img):
        conf = self._preprocess(img)
        inference_start = time.time()
        output = self.engine.infer(inputs={self.input_blob: conf['image']})
        inference_stop = time.time()

        bboxes, scores, classes = self._postprocess(output, conf)

        return bboxes, scores, classes, inference_stop-inference_start

    def _preprocess(self, img: np.ndarray):
        raise NotImplementedError

    def _postprocess(self, output, conf):
        raise NotImplementedError