# Implementation of Deep Convolutional Networks for Magnification of DICOM Brain Images
<br/> **Journal to be available online in February 2019 [here](https://scholar.google.com/citations?user=xxlAF58AAAAJ&hl=en#d=gs_md_cita-d&p=&u=%2Fcitations%3Fview_op%3Dview_citation%26hl%3Den%26user%3DxxlAF58AAAAJ%26citation_for_view%3DxxlAF58AAAAJ%3A2osOgNQ5qMEC%26tzom%3D480)**<br/>

*Convolutional Neural Networks have recently achieved great success in single image super-resolution (SISR). SISR is the action of 
reconstructing a high-quality image from a low-resolution one. In this paper, we propose a deep convolutional neural network (CNN) 
for the enhancement of Digital Imaging and Communications in Medicine (DICOM) brain images. The network learns an end-to-end mapping 
between the low and high resolution image. We first extract features from the image, where each new layer is connected to all previous 
layers. We then adopt residual learning and the mixture of convolutions to reconstruct the image. Our network is designed to work with 
grayscale images, since brain images are originally in grayscale. We further compare our method with previous works, trained on the same 
brain images, and show that our method outperforms them.* 

## Some Results from the Journal can be seen below
![1](https://user-images.githubusercontent.com/30661597/47485797-bea04780-d7f3-11e8-866f-5f5d955a60b1.PNG)

![2](https://user-images.githubusercontent.com/30661597/47485804-c3fd9200-d7f3-11e8-9baf-d792b4d10f9e.PNG)

![3](https://user-images.githubusercontent.com/30661597/47485812-c829af80-d7f3-11e8-8cba-cb42c1db97ed.PNG)

![4](https://user-images.githubusercontent.com/30661597/47485816-cb24a000-d7f3-11e8-942a-e26c1d8a1474.PNG)

## Requirements
This code was built in TensorFlow, Keras and MATLAB.

## Image Pre-Processing
Image resizing and data augmentation was done manually in MATLAB, for the purpose of a fair comparison with other papers. All images are then saved to a .h5 file, to be imported later during training

## Dataset
Please contact us for information about the dataset and it's acquisition. Note: The Dataset used has been acquired from specific hospitals.


