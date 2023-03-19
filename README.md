# Where's Waldo Detector

This repository contains scripts for detecting Waldo (from the "Where's Waldo?" series) in images using various image processing and computer vision techniques. It includes the following methods:

### 1. Template Matching

Template Matching performs template matching using the normalized cross-correlation method (cv2.TM_CCOEFF_NORMED). It then finds the best match's location and draws a rectangle around it in the scene image.



1. It works best when the template and the target in the scene are of the same size, orientation, and lighting conditions.
2. It can be sensitive to occlusions, rotations, and scale changes.


`template_matching.py` is a simple script that uses OpenCV's template matching technique to find Waldo in an image. This method works best when the template and the target in the scene have the same size, orientation, and lighting conditions.

### Usage
```
python template_matching.py
```
Make sure to update the `template_path` and `scene_path variables` in the script with the correct file paths for your template and scene images.

Keep in mind that this approach has some limitations, For more robust solutions, you might want to explore feature-based matching techniques like SIFT, SURF, or ORB, or even deep learning-based object detection and recognition methods.

### 2. ORB Feature Matching

The ORB feature detector and descriptor extractor to find keypoints and descriptors in both the template and scene images. It then uses the Brute Force Matcher with Hamming distance to match the descriptors. The script sorts the matches by distance and shows the top N matches in the scene image.

This approach is more robust than template matching, but it might still have some limitations, especially in cases of severe occlusions or extreme changes in scale or perspective. If you need even more robustness, consider using deep learning-based methods, such as convolutional neural networks (CNNs) trained for object detection or recognition tasks.

orb_matching.py uses the ORB (Oriented FAST and Rotated BRIEF) feature detector and descriptor extractor to match keypoints between the template and scene images. This method is more robust than template matching but may still have limitations in cases of severe occlusions or extreme changes in scale or perspective.

### Usage

```
python orb_matching.py
```
Make sure to update the template_path and scene_path variables in the script with the correct file paths for your template and scene images.

## 3. Deep Learning-based Object Detection (YOLOv5)

The deep learning-based approach requires several steps, including preparing a dataset, training a YOLOv5 model, and using the trained model to detect Waldo in images. Please refer to the instructions in the previous response for a detailed guide on how to use a YOLOv5 model for Waldo detection.

Using a deep learning-based object detection model like YOLO (You Only Look Once) or Faster R-CNN might provide better results.

To use a deep learning-based approach, you'll need a dataset of annotated images containing Waldo to train the model. If you don't have access to a dataset, you could use a pre-trained model and fine-tune it on a smaller dataset that you create. This process is known as transfer learning.

To get started with transfer learning using a pre-trained YOLO model, you can follow these steps:

Install the required libraries:
```
pip install tensorflow-gpu==2.5.0 opencv-python-headless
```
Clone the YOLOv5 repository:
```
git clone https://github.com/ultralytics/yolov5.git
```
```
cd yolov5
```
```
pip install -r requirements.txt
```
Prepare a dataset with annotated images of Waldo. Use a tool like VGG Image Annotator (VIA) to create annotations in YOLO format.

Split your dataset into training and validation sets. Organize the images and annotations in the following structure:
``` 
kotlin
data/
  train/
    images/
      image1.jpg
      image2.jpg
    labels/
      image1.txt
      image2.txt
  val/
    images/
      image3.jpg
      image4.jpg
    labels/
      image3.txt
      image4.txt
```
Create a YAML configuration file for your dataset:
```
# waldo_dataset.yaml
train: data/train
val: data/val
nc: 1  # number of classes
names: ['Waldo']
```
Choose a pre-trained YOLOv5 model and create a YAML configuration file for it. You can use the existing configurations in the yolov5/models/ folder. For example, you can use the YOLOv5s model:
```
# yolov5s_waldo.yaml
# parameters from yolov5s.yaml

# ...

# Change the number of classes to 1 and the depth_multiple and width_multiple to reduce the model size
nc: 1
depth_multiple: 0.33
width_multiple: 0.50

# ...
```
Train the YOLOv5 model on your dataset:
```
python train.py --img 640 --batch 8 --epochs 100 --data waldo_dataset.yaml --cfg yolov5s_waldo.yaml --weights yolov5s.pt --name yolov5s_waldo
```
After training, you can use the detect.py script in the YOLOv5 repository to find Waldo in your test images:
```
python detect.py --weights runs/train/yolov5s_waldo/weights/best.pt --img 640 --conf 0.25 --source test_images/
```
This process involves several steps and requires a dataset, but it is more likely to give you accurate and robust results compared to the ORB-based script. The deep learning model will learn to recognize Waldo's features and should be able to detect him even in challenging scenarios like the airport scene you mentioned.

### Usage

After training the YOLOv5 model on your dataset, you can use the detect.py script in the YOLOv5 repository to find Waldo in your test images:

```
python detect.py --weights runs/train/yolov5s_waldo/weights/best.pt --img 640 --conf 0.25 --source test_images/
```
Make sure to update the `--weights` and `--source` arguments with the correct file paths for your trained model and test images, respectively.


## License

### MIT License

