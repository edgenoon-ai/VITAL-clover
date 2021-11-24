import cv2

import numpy as np
import click

# from engines.yolox_engine import YOLOXInferenceEngine as Engine
# from engines.nanodet_engine import NanodetInferenceEngine as Engine
from engines.picodet_engine import PicodetInferenceEngine as Engine

import utils


@click.command()
@click.option('-p', '--model-path', type=str, required=True)
@click.option('-d', '--device', type=click.Choice(['CPU', 'GPU', 'MYRIAD'], case_sensitive=True), required=True)
@click.option('-i', '--image', type=str, required=True)
@click.option('-c', '--classes', type=int, required=True)
@click.option('--conf', type=float, required=True, default=0.3)
def main(model_path: str, device: str, image: str, classes: int, conf: float):
    infer = Engine(model_path=model_path, device=device, classes=classes)

    img = cv2.imread(image)
    b, s, c, t = infer(img=img)
    print(len(b))
    pred = utils.visualize(img=img, boxes=b, scores=s, cls_ids=c, conf=conf)
    cv2.imwrite(f'out_{image.split("/")[-1]}', pred)


if __name__ == "__main__":
    main()
