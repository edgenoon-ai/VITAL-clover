# VITAL-clover

The repository is related to [Neural and vision-based landing method](https://github.com/edgenoon-ai/clover/blob/neural_vision_based_landing_method/docs/en/neural_vision_based_landing_method.md) submission on [CopterHack 2022](https://clover.coex.tech/en/copterhack2022.html).

## Project idea
The project involves increasing the precision of landing (one of the most dangerous maneuvers for flying machines) under operational conditions on a mobile platform and taking care of safety in its vicinity.

Due to these reasons, we plan to implement autonomous landing on a specifically designed pad that is marked with graphical elements, making it possible to recover its relative pose and orientation. Additional safety measures will be implemented - no landing is attempted if persons are present in the vicinity of the landing pad. We want to achieve this using convolutional neural networks and USB inference accelerators, for example, Neural Compute Stick 2 or Google Coral USB Accelerator.


## Docker

We prepare a Dockerfile to start the PX4 Clover simulation with ROS Noetic. You can find it [here](./docker).

## Preliminary algorithm benchmarks

In the beginning, we want to compare some the-state-of-the-art algorithms for efficient object detection and measure their performance on Raspberry Pi4 with Intel Neural Computer Stick 2.

| Model                                                                                         	| Input resolution 	| COCO mAP 	| UAVVaste 1 class mAP 	|  Params [M] 	| Only inference latency [ms]	| Total inference latency [s]	|
|-----------------------------------------------------------------------------------------------	|:----------------:	|:--------:	|:------:	|:------:	|:---:	|:---:	|
| [YOLOv4-tiny](https://github.com/AlexeyAB/darknet)                                            	|      416*416     	|          	|        	|        	|     	|    	|
| [NanoDet-t](https://github.com/RangiLyu/nanodet)                                              	|      416*416     	|   23.5   	| 24.90 | 0.95       	| 0.0739 | 0.1402 |
| [NanoDet-m](https://github.com/RangiLyu/nanodet)                                              	|      416*416     	|   22.9   	| 24.30 | 3.81       	| 0.0720 | 0.1338  	|
| [PicoDet-S](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/picodet) 	|      416*416     	|   30.6   	| - |  0.99  |not compatible | not compatible   	| 
| [PicoDet-L](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/picodet) 	|      416*416     	|   36.6   	| - | 3.30         	| not compatible | not compatible   	|
| [YOLOX-Nano](https://github.com/Megvii-BaseDetection/YOLOX)                                   	|      416*416     	|   25.8   	| 22.65 | 0.91         	| 0.1133  | 0.1400    	|
| [YOLOX-Tiny](https://github.com/Megvii-BaseDetection/YOLOX)                                   	|      416*416     	|   32.8   	| 31.03 | 5.06         	| 0.1209 | 0.1405 |
