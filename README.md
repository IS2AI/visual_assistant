
# Visual Assistant for People with Visual Impairment

** Description **

## Computer Vision: A Deep Learning Approach for providing walking assistance to visually impaired


A brief description of what this project does and who it's for


### Prepraring the dataset

All of the models ([YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), 
[YOLOv7](https://github.com/WongKinYiu/yolov7)) were trained on the custom dataset which consisted of the original 
COCO 2017 dataset with additional annotations for faces created using YOLOv5-Face model. 

For this purpose, modifications were made to the `test_widerface.py` and the resultant 
script `coco_annotate_faces_custom.py` was saved in the corresponding github repo directory 
on the local machine (`../yolov5-face/`). Modifications mostly revolved around reading all 
the annotations currently present in labels txt files and re-writing them back into txt files but combined with face annotations. 

To reproduce this:
  1. Copy [yolov5-face github repo](https://github.com/deepcam-cn/yolov5-face) to your local machine
  2. Follow environment set up instructions provided 
  3. Download model weights you intend to use 
  (maximum complexity model preferred)

Once done, the script could be ran as follows:

```python
  python coco_annotate_faces_custom.py --weights 'yolov5-face model' 
  --conf-thres {float} --device {either cuda device (i.e. 0 or 1,2,3) or cpu}
  --augment --save_folder {Dir with existing txt annotations}
  --dataset_folder {COCO dataset path}

```
*--conf-thres 0.3 was used in our case*

### Model Training

When training and evaluating the models, all of the standard procedured outlined in the corresponding github repositories were followed:
* [YOLOv5 - Train Custom Model](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
* [YOLOv6](https://github.com/meituan/YOLOv6)
* [YOLOv7](https://github.com/WongKinYiu/yolov7)
 `nano` model version was chosen for each corresponding generation of YOLO family since it was intended for deployment on edge devices

### PyTorch to TensorRT Model Conversion 

NVIDIA TensorRT format allows for optimization for deep learning inference, delivering low latency and high throughput 
for inference application when deployed on embedded computing boards from Nvidia. 

Not all of the YOLO repository contain a direct deployment script for TensorRT format conversion. 
Fortunately, all models could be temporarily converted to ONNX format following instructions in each corresponding repository. Afterwards, 
ONNX to TensorRT converter provided by [Linaom1214](https://github.com/Linaom1214/TensorRT-For-YOLO-Series) can be used for every model generation. 

Converting to ONNX format:
* [YOLOv5 ONNX Export](https://github.com/ultralytics/yolov5/issues/251)
* [YOLOv6 ONNX Export](https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX)

In contrast, YOLOv7 repository already utilizes ONNX to TensorRT converter mentioned above and converts a model to the ONNX format all in one [Colab Notebook](https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7trt.ipynb).

[TensorRT-For-YOLO-Seriers](https://github.com/Linaom1214/TensorRT-For-YOLO-Series) repo created by Linaom1214 includes [Examples.ipynb](https://github.com/Linaom1214/TensorRT-For-YOLO-Series/blob/main/Examples.ipynb) Colab
 Notebook illustrating conversion for models of each YOLO family and should be followed with reference to the YOLOv7 example above. 

### Setting up Real-Sense D435 Camera

1. Install [Real-Sense SDK 2.0](https://dev.intelrealsense.com/docs/installation) for your operating system
2. Install [Python Wrapper](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python#installation) package by running `pip install pyrealsense2` 

There are Jupyter Notebooks for Intel RealSense SDK available in [librealsense](https://github.com/IntelRealSense/librealsense/tree/jupyter) github repo that showcase the minimal implementation of camera alignment and depth estimation. 

### Depth Estimation

There is an obvious problem when estimating the distance to the object using only the center point of the bounding box. 
In case of the sudden point conversion failure (depth value is zero), three-dimensional coordinate transformation 
will make mistakes, assigning the distance value with huge discrepancy. In order to make the algorithm more robust to similar cases Media Filter was implemented.
Several points with randomly initalized offset from the center of the bounding box are taken and the depth value for each of these points is estimated.
These points are then sorted based on the depth value and median filter is used to select the limited number of points for final distance estimation, which is evaluated by averaging the distance of the few select points.   

### Object Tracking

[Norfair](https://github.com/tryolabs/norfair) Python library was used to perform real-time 2D object tracking. 

 `Tracker` class imported from the library was leveraged to update the positions of the object based 
on their bounding boxes. A custom function `convert_detections_to_norfair_detections` was coded to store YOLO detections with depth (and/or color) information 
in `Detection` structure. Moreover, standard functions `draw_boxes` and `draw_tracked_objects` were modified to create a stylistically pleasing output. 

### Color Detection

In order to carry out the color detection, pre-processed initial frame has to be converted to HSV format. 

```python
 hsv_image = cv2.cvtColor(img.transpose((1,2,0)), cv2.COLOR_BGR2HSV)
```
*image has to transposed since the pre-processing step shifts the shape of an image, making number of color channels the first value in a tuple.

Custom `det_cropped_bboxes_colors` function was applied at the inference step to:
1. Crop a bounding box out of a frame
2. Split resulting cropped image into __h__,__s__,__v__ arrays
3. Compute mean values of __s__ and __v__ channels using `np.mean()`
4. Compute average hue using [mean of circular quantities](http://mkweb.bcgsc.ca/color-summarizer/?faq#averagehue)
5. Run calculated values through pre-defined thresholds to determine color name


The resulting algorithm to detect a color of a detected object was found to be computationally 
sub-optimal, resulting in FPS loss. Thus, `BaseEngine` class (initialized in `utils_norfair.py` and from which `Predictor` class inherits everything in `trt_norfair.py`) was 
modified to include a `show_color_flag`. This flag can be used to enable the functionality of color detection upon request. 


## Run Locally

Activate conda environment

```bash
  conda activate yolov6_deepsort 
  # deepsort is irrevelant to the currrent implementation
```

Go to the project directory

```bash
  cd C:/Users/mukal/tensorrt-python/yolov6
```

Run the script

```bash
  python trt_norfair.py
```



