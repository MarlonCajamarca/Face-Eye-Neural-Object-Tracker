# Face-Eye Neural Object Tracker 

Custom Face and Eye tracking implemented with a custom FaceEye pre-trained YOLOv4 model, DeepSort, on TensorFlow. YOLOv4 is a state of the art algorithm that uses deep convolutional neural networks to perform object detections. 

We can take the output of YOLOv4 and fed these object detections into Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric) in order to create a highly accurate object tracker.

## Getting Started
To get started, install the proper dependencies either via Anaconda or Pip. I recommend Anaconda route for people using a GPU as it configures CUDA toolkit version for you.

### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Pip
(TensorFlow 2 packages require a pip version >19.0.)
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```
### Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2

## Running the Face-Eye Tracker with custom YOLOv4
To run the Face-Eye object tracking using python scripts, all we need to do is run the object_tracker.py script to run the entire FaceEye tracking pipeline  using a custom YOLOv4 model.

In order to use the custom Face-eye Yolov4 provided for this project, first you need to create a folder called ``datasets`` in the root of the project (i.e. where you clones the project) and then download the provided open-source pre-trained model [HERE!](https://drive.google.com/file/d/1kRgU6tup_h67w8wUtQabgKVfn-eM4eYb/view?usp=sharing). Finally, you must unzip the file and that is all, you are ready to use the face-eye object tracker without the need for training a new model.

```bash
# Run yolov4 deep sort face-eye object tracker on test video 
python object_tracker.py --video ./inputs/input_video.mp4 --output ./outputs/output_video.mp4 --output_format mp4v --count True

# Run yolov4 deep sort face-eye object tracker on webcam (set video flag to 0)
python object_tracker.py --video 0 --output ./outputs/output_video.mp4.mp4 --output_format mp4v --count True

```
The output flag allows you to save the resulting video of the object tracker running so that you can view it again later. Video will be saved to the path that you set. (outputs folder is where it will be if you run the above command!)

If ``--count`` flag is active, the number of counted objects will be displayed in the live stream as well as in output video

## Resulting Video
As mentioned above, the resulting video will save to wherever you set the ``--output`` command line flag path to. I always set it to save to the 'outputs' folder. You can also change the type of video saved by adjusting the ``--output_format`` flag, by default it is set to mp4 codec which is mp4v.

## Filter Classes that are Tracked by Object Tracker
By default the code is setup to track all 2 classes (eye, face) classes from the coco dataset, which is what the custom FaceEye pre-trained YOLOv4 model is trained on. However, you can easily adjust a few lines of code in order to track any 1 or combination of the 2 classes. It is super easy to filter only the ``eye`` class or only the ``face`` class which are most common.

To filter a custom selection of classes all you need to do is comment out line 159 and uncomment out line 162 of [object_tracker.py](https://github.com/theAIGuysCode/yolov4-deepsort/blob/master/object_tracker.py) Within the list ``allowed_classes`` just add whichever classes you want the tracker to track. The classes can be any of the 2 that the model is trained on, see which classes you can track in the file at ./data/classes/FaceEyeTracker.names

## Command Line Args Reference

```bash
 object_tracker.py:
  --video: path to input video (use 0 for webcam)
    (default: './data/video/test.mp4')
  --output: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
    (default: None)
  --output_format: codec used in VideoWriter when saving video to file
    (default: 'mp4g)
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './datasets/EyeTracker_tf_model')
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
  --size: resize images to
    (default: 416)
  --iou: iou threshold
    (default: 0.45)
  --score: confidence threshold
    (default: 0.50)
  --dont_show: dont show video output
    (default: False)
  --info: print detailed info about tracked objects
    (default: False)
  --count: count objects being tracked on screen
    (default: False)
```

## Running FaceEyeTracker on Windows OS

If you want to run the object tracker from a webcam using a Windows OS, you only need to double click the **webcam_object_tracker.exe** application inside **WINDOWS folder**.

## Utility Tool for generating Yolo training artifacts

The project comes with a python script for creating all the yolo training artifacts used for training, i.e. ``obj.data``, ``train.txt`` and ``validation.txt`` partitions from a folder of annotated images and the ``configuration.json`` file provided.
If you want to trained or fine tune the pre-trained provided model, you can use this script to generate the artifacts and use transfer learning on your own custom dataset.

### Usage

```bash
 Create yolo-type artifacts for further transfer learning on your own custom dataset  
    
    python darknet_data.py  parent_folder  output_folder  config_file_path

 Positional arguments:
    parent_folder: Path to folder containing images and corresponding annotations in Yolo format
    output_folder: Path to folder where all the dataset and additional training artifacts will be stored
    config_file_path: Path to configuration.json file. In this file you configure the classes you will be used to train the yolo detector
