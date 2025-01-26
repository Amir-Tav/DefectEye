# DefectEye
Using Yolov8 and Raspberry pi to do some cool things!



yeee babbyyyy we out heree



# Steps:

Step 1: Dataset Preparation
Download the Dataset:

Visit the dataset page: 3Dprinting Object Detection Dataset (Roboflow).
Export the dataset in the format compatible with YOLOv5 (e.g., YOLO Darknet TXT format).
Organize Data:

Ensure the dataset is organized with images and corresponding label files. Each label file contains class names and bounding box coordinates.
Verify and Explore:

Inspect the images and annotations using tools like LabelImg or Roboflow’s built-in tools.
Ensure the dataset has diverse examples (variety of errors, lighting conditions, and printer setups).
Augment the Data:

Perform data augmentation to increase diversity:
Random rotations, flips, and scaling.
Brightness/contrast adjustments.
Noise addition.
You can use tools like Albumentations or Roboflow for augmentation.
