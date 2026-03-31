YOLOv26 re-implementation using PyTorch

### Installation

```
conda create -n YOLO python=3.10.10
conda activate YOLO
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python
pip install PyYAML
pip install tqdm
```

### Train

* Configure your dataset path in `main.py` for training
* Run `bash main.sh $ --train` for training, `$` is number of GPUs

### Test

* Configure your dataset path in `main.py` for testing
* Run `python main.py --test` for testing

### Results

| Version | Epochs | Box mAP |                                                                              Download |
|:-------:|:------:|--------:|--------------------------------------------------------------------------------------:|
|  v26_n  |  600   |    38.1 |                                                            [Model](./weights/best.pt) |
| v26_n*  |   -    |    40.2 | [Model](https://github.com/jahongir7174/YOLOv26-pt/releases/download/v0.0.1/v26_n.pt) |
| v26_s*  |   -    |    47.6 | [Model](https://github.com/jahongir7174/YOLOv26-pt/releases/download/v0.0.1/v26_s.pt) |
| v26_m*  |   -    |    52.3 | [Model](https://github.com/jahongir7174/YOLOv26-pt/releases/download/v0.0.1/v26_m.pt) |
| v26_l*  |   -    |    53.9 | [Model](https://github.com/jahongir7174/YOLOv26-pt/releases/download/v0.0.1/v26_l.pt) |
| v26_x*  |   -    |    56.6 | [Model](https://github.com/jahongir7174/YOLOv26-pt/releases/download/v0.0.1/v26_x.pt) |

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.381
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.541
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.410
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.190
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.422
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.540
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.318
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.533
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.591
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.380
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.648
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.769
```

* `*` means that it is from original repository, see reference
* In the official YOLOv26 code, mask annotation information and Objects365 dataset pretrained weights are used, which
  leads to higher performance

### Dataset structure

    ├── COCO 
        ├── images
            ├── train2017
                ├── 1111.jpg
                ├── 2222.jpg
            ├── val2017
                ├── 1111.jpg
                ├── 2222.jpg
        ├── labels
            ├── train2017
                ├── 1111.txt
                ├── 2222.txt
            ├── val2017
                ├── 1111.txt
                ├── 2222.txt

#### Reference

* https://github.com/ultralytics/ultralytics
* https://github.com/jahongir7174/YOLOv8-pt
