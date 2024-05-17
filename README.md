# Deep Learning model based on convolutional neural networks to improve breast cancer classification implemented on Pytorch 

In this repository, I implemented the deep learning classifier introduced in the [paper](https://www.nature.com/articles/s41598-019-48995-4) "Deep Learning to Improve Breast Cancer Detection on Screening Mammography" using PyTorch.  The original code and model are available [here](https://github.com/lishen/end2end-all-conv). However, this  code is in Keras.  
My  main goal is to provide a  comprehensible implementation of this model, which can be helpful for everyone, especially those who are beginning to work with deep learning and are interested in medical applications.   

## Summary of the main contribution of this paper

The authors propose a breast cancer classifier based on a methodology composed of 2 stages: The first stage consists of a **patch-level classifier** that uses pixel-level annotations from the mammograms to discriminate the regions of interest and train the model only based on those areas. The second stage consists of a **whole image classifier**. This image classifier uses the patch classifier as a backbone, removing only the top layers from the patch classifier while incorporating two additional layers. The training of this whole image classifier  requires only image-level labels. I describe the patch level and the whole image classifiers in more detail as follows: 


### Patch-Level Classifier

#### Patch Dataset 
We generate two datasets from all the mammograms. The first dataset (S) consists of one patch extracted from the center of the ROI and another background patch randomly sampled from the same image. The second dataset (s10) consists of 20 patches:  10 patches randomly selected from each ROI, with a minimum overlapping ratio of 0.9, plus 10 patches randomly selected from anywhere in the image other than the ROI. All patches have the size of 224*224 and are saved as 16-bit PNG files. Additionally, the patches are divided into one of the five classes: 0: Background, 1: Malignant Calcification, 2: Benign Calcification, 3: Malignant Mass, and 4: Benign Mass.
We must remove the mammograms' watermarks before extracting the patches and rescale the pixel values to [0.0,1.0].

##### Remove watermarks from mammograms by using the segment_breast function located /Patches/tools.py
##### Generate patches from the mammograms by using the generatin_patches.py script in the "patches" folder.


#### Preprocessing 
We convert mammograms from DICOM files into 16-bit PNG files. Then, we resize the mammograms to 1152*896  pixels. There is no cropping or reorienting of the mammograms. We split  the dataset  into training and test sets using an 85/15 % split. We further split the training set to generate a validation set using a 90/10 % split.   The partitions  are  stratified to maintain the same  proportion of cancer cases across all sets. 

#### MODEL 
ResNet50

#### Training Strategy 

The ResNet50 is trained in three stages. All learning parameters are freezing in the first stage except those in the final layer. Then, layers are gradually unfrozen from top to bottom. At the same time, the learning rate is decreased in each stage. The 3-stage training method on S and S10 datasets is as follows:
1. **First Stage**: Set the learning rate to 1e-3, weight decay to  1e-4, and train only the fully connected layer for three epochs.
2. **Second Stage**: Set the learning rate to 1e-4, weight decay to  1e-4, and train the last three convolutional neural layers and the fully connected layer for ten epochs. <ins> According to the Pytorch notation, these layers correspond to Layer 4.2 and FC </ins>
3. **Third Stage**: Set Learning rate to  1e-5 and train all layers for 37 epochs

During training, we augment mammograms to promote model generalizability by applying the following augmentations:
- Horizontal and vertical flips 
- Rotations in [-25,25] degrees
- Zoom in [0.8,1.2] ratio
- Intensity shift in [-20,20] % of pixel values
- Shear in [-12,12] grades

We train the Resnet50 for 50 epochs in total. However, since the S dataset is much smaller than s10, we increase the number of epochs in the third stage to 100. The batch size is 256, and we use ADAM as the optimizer.  The model's parameters are initialized with the pre-trained weights in Imagenet.

##### Train patch classifier by using train_function.py in "patch_classifier" folder 
##### Trained Patch-level classifier models are in trained_models 

| Dataset      | Validation Acc.| Test Acc.     |
| :---         |     :---:      |          ---: |
| s            | 0.800          | 0.812         |
| s10          | 0.970          | 0.967         |

### Whole image Classifier 
According to the configurations tested in the [paper](https://www.nature.com/articles/s41598-019-48995-4) to convert the patch classifier to a whole Image classifier, the design with the best performance corresponds to the Resnet50 classifier followed by two identical Resnet blocks of [512-512-1024]. Resnet blocks consist of repeated units of three convolutional layers with filter sizes 1x1, 3x3, and 1x1. Therefore, the numbers in the brackets  indicate the depths of the three convolutional layers in each block. Before assembling the Resnet blocks in the patch classifier, the fully connected layer is replaced by a Global Average Pooling, which outputs the average activation of each feature map (there are 2048 feature maps in the last convolutional layer for Resnet50).  We connect the two Resnet blocks to a fully connected layer that predicts one of the classes we want to classify: benign and malignant.

##### Image Classifier model is defined in whole_classifier_model.py located in  the "whole_image_classifier" folder

Similarly to the training method used for the patch classifier, we employ a 2-stage training strategy for the whole image classifier, which  is as follows:

1. **First Stage**: Set the learning rate to 1e-4, weight decay to 1e-3, and train only the newly added layers to the model for 30 epochs.
2. **Second Stage**: Set the Learning rate to 1e-5 and train all layers for 20 epochs.

##### The script to train the whole image classifier is in /whole_image_classifier/main.py

Due to the GPU memory limit, we decreased the batch size to 12. We optimized the model with Adam and used the same augmentations applied  in the patch classification.  

**Note**: The backbone used in the whole image classifier corresponds to the ResNet50 trained on the s10 patch dataset. 

| Model                    | Validation Acc.| Test Acc.     |
| :---                     |     :---:      |          ---: |
| ResNet50+2 ResNet Blocks |     0.857      | 0.856         |
                                




