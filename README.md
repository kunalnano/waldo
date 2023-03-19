# Where's Waldo Detector

This repository contains scripts for detecting Waldo (from the "Where's Waldo?" series) in images using various image processing and computer vision techniques. It includes the following methods:

1. Template Matching
2. ORB Feature Matching
3. Deep Learning-based Object Detection (YOLOv5)

## 1. Template Matching

`template_matching.py` is a simple script that uses OpenCV's template matching technique to find Waldo in an image. This method works best when the template and the target in the scene have the same size, orientation, and lighting conditions.

### Usage
```
python template_matching.py
```
Make sure to update the template_path and scene_path variables in the script with the correct file paths for your template and scene images.

## 2. ORB Feature Matching

orb_matching.py uses the ORB (Oriented FAST and Rotated BRIEF) feature detector and descriptor extractor to match keypoints between the template and scene images. This method is more robust than template matching but may still have limitations in cases of severe occlusions or extreme changes in scale or perspective.

### Usage

```
python orb_matching.py
```

Make sure to update the template_path and scene_path variables in the script with the correct file paths for your template and scene images.

## 3. Deep Learning-based Object Detection (YOLOv5)

The deep learning-based approach requires several steps, including preparing a dataset, training a YOLOv5 model, and using the trained model to detect Waldo in images. Please refer to the instructions in the previous response for a detailed guide on how to use a YOLOv5 model for Waldo detection.

### Usage

After training the YOLOv5 model on your dataset, you can use the detect.py script in the YOLOv5 repository to find Waldo in your test images:

```
python detect.py --weights runs/train/yolov5s_waldo/weights/best.pt --img 640 --conf 0.25 --source test_images/
```

Make sure to update the --weights and --source arguments with the correct file paths for your trained model and test images, respectively.


## License

### MIT License

