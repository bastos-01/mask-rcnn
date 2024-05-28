# Mask R-CNN for Object Detection and Instance Segmentation on Keras and TensorFlow 2.14.0 and Python 3.10.12
This is an implementation of the [Mask R-CNN](https://arxiv.org/abs/1703.06870) paper which edits the original [Mask_RCNN](https://github.com/matterport/Mask_RCNN) repository (which only supports TensorFlow 1.x), so that it works with Python 3.10.12 and TensorFlow 2.14.0. This new reporsitory allows to train and test (i.e make predictions) the Mask R-CNN  model in TensorFlow 2.14.0. The Mask R-CNN model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

The implementation shown in `samples/segmentation` is to train a model with a dataset containing historical Portuguese documents, in the `PAGE XML` format, following the example in [Balloon Example](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46).

Compared to the source code of the old [Mask_RCNN](https://github.com/matterport/Mask_RCNN) repo, the  edits the following 2 modules:

1. `model.py`
2. `utils.py`

Apart from that, this repository uses the same training and testing code as in the old repo and similarly includes:

* Source code of Mask R-CNN built on FPN and ResNet101.
* Jupyter notebooks to visualize the detection pipeline at every step
* Example of training on your own dataset located at `samples/segmentation`

## Requirements
The [Mask-RCNN_TF2.14.0](https://github.com/z-mahmud22/Mask-RCNN_TF2.14.0) repo is tested with TensorFlow 2.14.0, Keras 2.14.0, and Python 3.10.12 for the following system specifications:

1. GPU - `GeForce RTX 3060 8GiB`
2. OS -  `Ubuntu20.04`

Other common packages required for this repo are listed in `requirements.txt` and `environment.yml`.

## Installation
**Recommended way:**

1. Clone this repository
   ```bash
   git clone https://github.com/bastos-01/kraken.git maskrcnn
   ```

2. Create environment with anaconda and install dependencies:
   ```bash
   conda env create -f environment.yml 
   ```
   
**Alternative way:**

1. Clone this repository
   ```bash
   git clone https://github.com/bastos-01/kraken.git maskrcnn
   ```
   
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ```


# Getting Started

* [model.py](mcnn/model.py), [utils.py](mcnn/utils.py), [config.py](mcnn/config.py): These files contain the main Mask RCNN implementation.

* [inspect_segmentation_data.ipynb](samples/segmentation/inspect_segmentation_data.ipynb): This notebook visualizes the different pre-processing steps to prepare the training data.

* [inspect_segmentation_model.ipynb](samples/segmentation/inspect_segmentation_model.ipynb): This notebook goes in depth into the steps performed to detect and segment objects. It provides visualizations of every step of the pipeline.

* [segmentation.py](samples/segmentation/segmentation.py): This python script is an adapted version of the file `samples/coco/coco.py` that overrides the needed classes and functions to train my dataset.


# Training on Your Own Dataset

Start by reading this [blog post about the balloon color splash sample](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46). It covers the process starting from annotating images to training and then using the results in a sample application.

In summary, to train the model on your own dataset, you'll need to extend two classes:

```Config```
This class contains the default configuration. Subclass it and modify the attributes you need to change.

```Dataset```
This class provides a consistent way to work with any dataset. 
It allows you to use new datasets for training without having to change 
the code of the model. It also supports loading multiple datasets at the
same time, which is useful if the objects you want to detect are not 
all available in one dataset. 

## Differences from the Official Paper
This implementation follows the [Mask RCNN paper](https://arxiv.org/abs/1703.06870) for the most part, but there are a few cases where we deviated in favor of code simplicity and generalization. These are some of the differences we're aware of. If you encounter other differences, please do let us know.

* **Image Resizing:** To support training multiple images per batch we resize all images to the same size. For example, 1024x1024px on MS COCO. We preserve the aspect ratio, so if an image is not square we pad it with zeros. In the paper the resizing is done such that the smallest side is 800px and the largest is trimmed at 1000px.
* **Bounding Boxes**: Some datasets provide bounding boxes and some provide masks only. To support training on multiple datasets we opted to ignore the bounding boxes that come with the dataset and generate them on the fly instead. We pick the smallest box that encapsulates all the pixels of the mask as the bounding box. This simplifies the implementation and also makes it easy to apply image augmentations that would otherwise be harder to apply to bounding boxes, such as image rotation.

    To validate this approach, we compared our computed bounding boxes to those provided by the COCO dataset.
We found that ~2% of bounding boxes differed by 1px or more, ~0.05% differed by 5px or more, 
and only 0.01% differed by 10px or more.

* **Learning Rate:** The paper uses a learning rate of 0.02, but we found that to be
too high, and often causes the weights to explode, especially when using a small batch
size. It might be related to differences between how Caffe and TensorFlow compute 
gradients (sum vs mean across batches and GPUs). Or, maybe the official model uses gradient
clipping to avoid this issue. We do use gradient clipping, but don't set it too aggressively.
We found that smaller learning rates converge faster anyway so we go with that.

## Citation
Use this bibtex to cite this repository:
```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}

@INPROCEEDINGS{8237584,
  author={He, Kaiming and Gkioxari, Georgia and Dollár, Piotr and Girshick, Ross},
  booktitle={2017 IEEE International Conference on Computer Vision (ICCV)}, 
  title={Mask R-CNN}, 
  year={2017},
  volume={},
  number={},
  pages={2980-2988},
  doi={10.1109/ICCV.2017.322}}
```

## Contribution
Contributions to this repository are welcome. Examples of things you can contribute:
* Speed Improvements. Like re-writing some Python code in TensorFlow.
* Training on other datasets.
* Accuracy Improvements.
* Visualizations and examples.
* Update the TF-1 docker image to support TF-2 implementation

# Projects Using this Model
### [4K Video Demo](https://www.youtube.com/watch?v=OOT3UIXZztE) by Karol Majek.
[![Mask RCNN on 4K Video](assets/4k_video.gif)](https://www.youtube.com/watch?v=OOT3UIXZztE)

### [Images to OSM](https://github.com/jremillard/images-to-osm): Improve OpenStreetMap by adding baseball, soccer, tennis, football, and basketball fields.
![Identify sport fields in satellite images](assets/images_to_osm.png)

### [Splash of Color](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46). A blog post explaining how to train this model from scratch and use it to implement a color splash effect.
![Balloon Color Splash](assets/balloon_color_splash.gif)

### [Segmenting Nuclei in Microscopy Images](samples/nucleus). Built for the [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018)
Code is in the `samples/nucleus` directory.

![Nucleus Segmentation](assets/nucleus_segmentation.png)

### [Detection and Segmentation for Surgery Robots](https://github.com/SUYEgit/Surgery-Robot-Detection-Segmentation) by the NUS Control & Mechatronics Lab.
![Surgery Robot Detection and Segmentation](https://github.com/SUYEgit/Surgery-Robot-Detection-Segmentation/raw/master/assets/video.gif)

### [Reconstructing 3D buildings from aerial LiDAR](https://medium.com/geoai/reconstructing-3d-buildings-from-aerial-lidar-with-ai-details-6a81cb3079c0)
A proof of concept project by [Esri](https://www.esri.com/), in collaboration with Nvidia and Miami-Dade County. Along with a great write up and code by Dmitry Kudinov, Daniel Hedges, and Omar Maher.

![3D Building Reconstruction](assets/project_3dbuildings.png)

### [Usiigaci: Label-free Cell Tracking in Phase Contrast Microscopy](https://github.com/oist/usiigaci)
A project from Japan to automatically track cells in a microfluidics platform. Paper is pending, but the source code is released.

![](assets/project_usiigaci1.gif) ![](assets/project_usiigaci2.gif)

### [Characterization of Arctic Ice-Wedge Polygons in Very High Spatial Resolution Aerial Imagery](http://www.mdpi.com/2072-4292/10/9/1487)
Research project to understand the complex processes between degradations in the Arctic and climate change. By Weixing Zhang, Chandi Witharana, Anna Liljedahl, and Mikhail Kanevskiy.

![image](assets/project_ice_wedge_polygons.png)

### [Mask-RCNN Shiny](https://github.com/huuuuusy/Mask-RCNN-Shiny)
A computer vision class project by HU Shiyu to apply the color pop effect on people with beautiful results.

![](assets/project_shiny1.jpg)

### [Mapping Challenge](https://github.com/crowdAI/crowdai-mapping-challenge-mask-rcnn): Convert satellite imagery to maps for use by humanitarian organisations.
![Mapping Challenge](assets/mapping_challenge.png)

### [GRASS GIS Addon](https://github.com/ctu-geoforall-lab/i.ann.maskrcnn) to generate vector masks from geospatial imagery. Based on a [Master's thesis](https://github.com/ctu-geoforall-lab-projects/dp-pesek-2018) by Ondřej Pešek.
![GRASS GIS Image](assets/project_grass_gis.png)
