# Object Localization Data Prep 

## Dataset 

- [50 Dogs Images](https://github.com/myselfHimanshu/ai-vision-program/tree/master/Session-12/DataPrep/images)

## Annotations Creation 

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