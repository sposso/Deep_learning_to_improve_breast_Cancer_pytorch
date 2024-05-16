# Deep Learning model based on convolutional neural networks to improve breast cancer classification implemented on Pytorch 

In this repository, I implemented the deep learning classifier introduced in the [paper](https://www.nature.com/articles/s41598-019-48995-4) "Deep Learning to Improve Breast Cancer Detection on Screening Mammography" using PyTorch.  The original code and model are available [here](https://github.com/lishen/end2end-all-conv). However, this implementation is in Keras. 


## Summary of the main contribution of this paper

The authors propose a breast cancer classifier based on a methodology composed of 2 stages: The first stage consists of a **patch-level classifier** that uses pixel-level annotations from the mammograms to discriminate the regions of interest and train the model only based on those areas. The second stage consists of a **whole image classifier**. This image classifier uses the patch classifier as a backbone, removing only the top layers from the patch classifier while incorporating two additional layers. The training of this whole image classifier  requires only image-level labels. I describe the patch level and the whole image classifiers in more detail as follows: 

### Patch-Level Classifier

#### Patch Dataset 
We generate two datasets from all the mammograms. The first dataset (S) consists of one patch extracted from the center of the ROI and another background patch randomly sampled from the same image. The second dataset (s10) consists of 20 patches:  10 patches randomly selected from each ROI, with a minimum overlapping ratio of 0.9, plus 10 patches randomly selected from anywhere in the image other than the ROI. All patches have the size of 224*224 and are saved as 16-bit PNG files. Additionally, the patches are divided into one of the five classes: 0: Background, 1: Malignant Calcification, 2: Benign Calcification, 3: Malignant Mass, and 4: Benign Mass.
It is important to note that before extracting the patches, we remove the mammograms' watermarks and rescale the pixel values to [0.0,1.0].

#### MODEL 
ResNet50

#### Training Strategy 

The ResNet50 is trained in three stages. All learning parameters are freezing in the first stage except those in the final layer. Then, layers are gradually unfrozen from top to bottom. At the same time, the learning rate is decreased in each stage. The 3-stage training method on S and S10 datasets is as follows:
1. **First Stage**: Set the learning rate to 1e-3, weight decay to  1e-4, and train only the fully connected layer for three epochs.
2. **Second Stage**: Set the learning rate to 1e-4, weight decay to  1e-4, and train the last three convolutional neural layers and the fully connected layer for ten epochs. <ins> According to the Pytorch notation, these layers correspond to Layer 4.2 and FC </ins>

