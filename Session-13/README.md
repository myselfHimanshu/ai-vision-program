# Session 13 - YOLOv2 and YOLOv3

## Objectives

- Obj-1
    - Use OpenCV YOLO.
    - Take an image of an object which is there in COCO data set. 
    - Run this image through the opencv code. 
    - Show annotated image by YOLO. 

- Obj-2
    - Training Custom Dataset for YOLOv3. 
    - Collect a dataset of 500 images and annotate them.
    - Download video from youtube which has above class. 
    - Use ffmpeg to extract frames from the video. 
    - Inter on these images using detect.py file.
    - Use ffmpeg to convert the files in your output folder to video
    - Upload the video to YouTube. 
    - Share the link of your YouTube video on LinkedIn or Instagram.

## RESULTS FOR Objective 1

[Objective 1](https://github.com/myselfHimanshu/ai-vision-program/tree/master/Session-13/opencv-detection)

## RESULTS FOR Objective 2

### Output a single frame from the video into an image file:
ffmpeg -i input.mov -ss 00:00:14.435 -vframes 1 out.png

### Output one image every second, named out1.png, out2.png, out3.png, etc.
### The %01d dictates that the ordinal number of each output image will be formatted using 1 digits.
ffmpeg -i input.mov -vf fps=1 out%d.png

### Output one image every minute, named out001.jpg, out002.jpg, out003.jpg, etc. 
### The %02d dictates that the ordinal number of each output image will be formatted using 2 digits.
ffmpeg -i input.mov -vf fps=1/60 out%02d.jpg

### Extract all frames from a 24 fps movie using ffmpeg
### The %03d dictates that the ordinal number of each output image will be formatted using 3 digits.
ffmpeg -i input.mov -r 24/1 out%03d.jpg

### Output one image every ten minutes:
ffmpeg -i input.mov -vf fps=1/600 out%04d.jpg