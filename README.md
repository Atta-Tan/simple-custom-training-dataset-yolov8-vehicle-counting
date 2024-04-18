# vehicle-classificataion-and-counting-system
This repository is a simple guidance of counting and classification system for vechicle object using YOLOv8 model, developed by Ultralytics team.
YOLOv8 is one of popular real-time detection and classification tasks and supports multi-object tracking (MOT) algorithms like Bot-SORT and ByteTrack also. This model is very friendly for new users and powerful one.
For counting vehicle objects, YOLOv8 was trained by using COCO128 datasets, so YOLOv8 training wieght, internal parameters, can be used to detect and classify any vehicle classes in COCO128 dataset such as car, motorcycle, bus, and truck easily and accurately. 
But if we want to use it for any custom classes, we need to train our custom training weight first. This model also supports training mode that we can train any custom datasets for some specific object detection and classification.

In this project, I will show how to easily setup process and run counting py file using YOLOv8 first, and the next step, I will display how to train custom data weight for my specific task, to classify 11 different vehicle classes.

more information about Ultralytics: https://docs.ultralytics.com/

## Getting Started
clone this repo to get .py files for vehicle connting
```sh
git clone https://github.com/Atta-Tan/simple-custom-training-dataset-yolov8-vehicle-counting.git
```
next, create virtual environment and install ultralytics (or clone Ultralytics repo (optional))

```sh
cd ...ROOTPATH.../simple-custom-training-dataset-yolov8-vehicle-counting
python -m venv ENV
pip install Ultralytics
```
Activate virtual environment (as ENV) and install packages

```sh
.\ENV\Scripts\activate
pip install -e .
```

(Optional) In case of need to run with GPU
re-install pytorchon our machine
```sh
pip3 uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

more information about Pytorch: https://pytorch.org/get-started/locally/

Now our environment is ready!!

## Running counting vehicles (4 classes in COCO128 Dataset)
Setting INPUT_PATH, OUTPUT_PATH, WEIGHT_PATH (eg. yolov8m.pt)
Then, run py file

```sh
python ./py_files/yolov8_custom_counting_tocsvfile.py
```
![Running Process](images/run_to_count.png)



