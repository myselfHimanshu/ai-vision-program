# Session 12 - OBJECT LOCALIZATION : YOLO

## Objectives

- Obj-1
    - Download this TINY [IMAGENET](http://cs231n.stanford.edu/tiny-imagenet-200.zip) dataset. 
    - Train ResNet18 on the dataset (70/30 split) for 50 Epochs. 
    - Target 50%+ Validation Accuracy. 

- Obj-2
    - Download 50 images of dogs. 
    - Use this [tool](http://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html) to annotate bounding boxes around the dogs.
    - Download JSON file. 
    - Describe the contents of this JSON file in FULL details.
    - Find out the best total numbers of clusters.

## MAIN CODE

The code has been shifted to new repo to keep things clean in this repo.

Link to repo : [ULTRON-VISION](https://github.com/myselfHimanshu/ultron-vision/tree/session-12)

## RESULTS

- For one_cycle policy, div_factor = 10
    - initial learning rate : 0.01/10 = 0.001
- Annihilation, final_div_factor = 100
    - min learning rate : 0.0001

<p align="center">
  <img width="50%" height="50%" src="https://github.com/myselfHimanshu/ultron-vision/raw/session-12/experiments/tinyimagenet-exp-002/stats/lr.png"/>
</p>

The configuration for the experiment can be found in `config file`. The training and validation loss and accuracy logs can be found in `LOGS` file.

<b> Steps for data creation and preprocessing: </b>

- Download data using provided link.
- Read annots file of each class in train folder and annots file in val folder and store into one pandas dataframe with columns as:
    - image_name, class, x1, x2, x3, x4
- Shuffle Pandas Dataframe
- Use Sklearn train_test_split (70/30) split on the dataframe and save.
- Build CustomDataset class for reading data provided train or val csv files and respective transforms. 

- FILES
    - [CustomDataset Building FILE](https://github.com/myselfHimanshu/ultron-vision/blob/session-12/infdata/dataset/tinyimagenet_data.py)
    - [Config FILE](https://github.com/myselfHimanshu/ultron-vision/blob/session-12/experiments/tinyimagenet-exp-002/summaries/config.txt)
    - [Albumenation transformation](https://github.com/myselfHimanshu/ultron-vision/blob/session-12/infdata/transformation/tinyimagenet_tf.py)
    - [TinyImageNet agent](https://github.com/myselfHimanshu/ultron-vision/blob/session-12/agents/tinyimagenet_agent.py)
    - [TinyImageNet inference agent](https://github.com/myselfHimanshu/ultron-vision/blob/session-12/inference/tinyimagenet_iagent.py)
    - [LOGS FILE](https://github.com/myselfHimanshu/ultron-vision/blob/session-12/experiments/tinyimagenet-exp-002/logs/exp_debug.log)

- Best Validataion Accuracy : 58.35%
- Total Epochs : 30

## Loss and Accuracy Graph

|Loss|Accuracy|
|--|--|
|<p align="center"><img width="80%" height="80%" src="https://github.com/myselfHimanshu/ultron-vision/raw/session-12/experiments/tinyimagenet-exp-002/stats/loss.png"/></p>|<p align="center"><img width="80%" height="80%" src="https://github.com/myselfHimanshu/ultron-vision/raw/session-12/experiments/tinyimagenet-exp-002/stats/accuracy.png"/></p>|

## Misclassified Images with Gradcam

True and predicted values are written on top of images (can be quite small font in here, just click it to open in new tab)

<p align="center">
  <img width="50%" height="50%" src="https://github.com/myselfHimanshu/ultron-vision/raw/session-12/experiments/tinyimagenet-exp-002/stats/misclassified_imgs.png"/>
</p>

# Object Localization Data Prep 

## Dataset 

- [50 Dogs Images](https://github.com/myselfHimanshu/ai-vision-program/master/Session-12/DataPrep/images/)

## Bounding Box Creation 

- [Annotations File](https://github.com/myselfHimanshu/ai-vision-program/blob/master/Session-12/DataPrep/annotations.json)

This is an instance from annotations file :

- Axes format for image
    - top-left coordinate of image : (0,0)
    - bottom-right coordinate of image : (width_of_image, height_of_image)

```
'dog001.jpg77283': {'filename': 'dog001.jpg',
  'size': 77283,
  'regions': [{'shape_attributes': {'name': 'rect',
     'x': 263,
     'y': 30,
     'width': 112,
     'height': 103},
    'region_attributes': {'animal': 'dog'}}],
  'file_attributes': {'caption': '', 'public_domain': 'no', 'image_url': ''}},
```

- 'filename' : filename of image
- 'regions' : 'x' : top_left corner x-coordinate of bouding box
- 'regions' : 'y' : top_left corner y-coordinate of bouding box
- 'regions' : 'width' : width of bouding box
- 'regions' : 'height' : height of bouding box
- 'region_attributes' : 'animal' is parent class, 'dog' is the class of bounding box

## Finding number of achor boxes to use using KMeans is explained in jupyter notebook

- [Main File](https://github.com/myselfHimanshu/ai-vision-program/blob/master/Session-12/DataPrep/detection.ipynb)


