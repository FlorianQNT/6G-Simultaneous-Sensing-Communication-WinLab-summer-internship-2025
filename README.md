# 6G_Simultaneous_Sensing_Communication

This project was made in the context of the summer internship in WinLab, Rutgers in 2025.

Intern:<br>
-Florian QUINT<br>

Advisor:<br>
-Ivan SESKAR<br>
-professor Narayan MANDAYAM<br>

The goal of this project is to be able to detect and classify object while using a camera and/or a radar.<br>

# We started by doing it only with a camera.<br>
For this, we used the Deep-Learning algorithm called YOLO (You Only Look Once).<br>
YOLO was firstly created by Joseph Redmaon in 2015. He made 3 version.<br>
The versions that came after are made by other researchers and we can find all version after YOLOv3 on ultralytics website.<br>
The version we are using is YOLOv8 witch is available since january 2023.<br>
This algorithm is trained on multiple dataset and is considered one of the best for real time object detection.<br>

With the camera, we focused on detecting and classifying human and chair although it would work with every class from COCO dataset.<br>
The camera that we used is a ZED 2i. It was on a desktop with those specs:<br>
-CPU:Intel Core i9-14900KF 3.20GHz<br>
-GPU:NVIDIA GeForce RTX 4080 SUPER 16Gb<br>
-RAM:DDR4 32Gb 5600MHz<br>
-Windows11 Pro<br>

We did another version of the program so it could work on a laptop using the integrated camera.<br>
It was tried and worked on a laptop with those specs:<br>
-CPU: Intel Core i5-8250U 1.80GHz<br>
-GPU: NVIDIA GeForce GTX 1050 2Gb<br>
-RAM: DDR4 16Gb 2400MHz<br>
-Windows11 Home<br>

The biggest difference between both version is that thanks to the ZED camera, we can have the distance between the object and the camera.<br>

We tried with another dataset named OpenImagesv7. It works and has more class on yolo but it also impact the confidence for the classes.<br>
For the same objects at the same place with both dataset, OpenImages was roughly 15% less confident.<br>
This can be explained by the difference between the number of classes on both dataset:<br>
COCO has 80 classes and OpenImagesv7 has 600 classes.<br>

# To continue this project, we added a radar<br>
The radar that we used is an IBM 28GHz phased-array antenna module<br>
For this radar, we are using the GR-mimo module made for GNU Radio.<br>
