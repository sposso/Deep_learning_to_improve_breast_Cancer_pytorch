# Deep Learning model based on convolutional neural networks to improve breast cancer classification implemented on Pytorch 

In this repository, I implemented the deep learning classifier introduced in the [paper](https://www.nature.com/articles/s41598-019-48995-4) "Deep Learning to Improve Breast Cancer Detection on Screening Mammography" using PyTorch.  The original code and model are available [here](https://github.com/lishen/end2end-all-conv). However, this  code is in Keras.  
My  main goal is to provide a  comprehensible implementation of this model, which can be helpful for everyone, especially those who are beginning to work with deep learning and are interested in medical applications.   

The mammography dataset employed in this study is the CBIS_DDSM. [Here](https://github.com/sposso/CBIS-DDSM-DATASET), you can find a short tutorial on setting up the data. 

# Summary of the main contribution of this paper

The authors propose a breast cancer classifier based on a methodology composed of 2 stages: The first stage consists of a **patch-level classifier** that uses pixel-level annotations from the mammograms to discriminate the regions of interest and train the model only based on those areas. The second stage consists of a **whole image classifier**. This image classifier uses the patch classifier as a backbone, removing only the top layers from the patch classifier while incorporating two additional layers. The training of this whole image classifier  requires only image-level labels. I describe the patch level and the whole image classifiers in more detail as follows: 

# Mammograms preprocessing: 

We convert mammograms from DICOM files into 16-bit PNG files. Then, we resize the mammograms to 1152*896  pixels. There is no cropping or reorienting of the mammograms. We split  the dataset  into training and test sets using an 85/15 % split. We further split the training set to generate a validation set using a 90/10 % split.   The partitions  are  stratified to maintain the same  proportion of cancer cases across all sets. 

# First stage:  Patch-Level Classifier

## Patch Dataset Generation from Mammograms

We generate two datasets from the mammograms for patch-based classification:

### 📁 Dataset S
- Contains **2 patches per image**:
  - 1 patch extracted from the **center of the Region of Interest (ROI)**.
  - 1 **background patch** randomly sampled from the same image (outside the ROI).

### 📁 Dataset s10
- Contains **20 patches per image**:
  - 10 patches **randomly sampled from each ROI** with at least **90% overlap**.
  - 10 patches **randomly sampled from non-ROI regions**.

### 🖼️ Patch Details
- **Size**: 224 × 224 pixels  
- **Format**: 16-bit PNG  
- **Pixel values**: Rescaled to the range **[0.0, 1.0]**
- **Watermarks** are removed before patch extraction.

### 🏷️ Class Labels
Each patch is assigned to one of the following five classes:

| Label | Description               |
|-------|---------------------------|
| 0     | Background                |
| 1     | Malignant Calcification   |
| 2     | Benign Calcification      |
| 3     | Malignant Mass            |
| 4     | Benign Mass               |

To generate both **S** and **s10** patch datasets simultaneously, navigate to the `patches` folder and run the following script:

```bash
cd patches
python generating_patches.py

```


# Patch Classification with ResNet50

The chosen model for patch classification is **ResNet50**, trained using a **3-stage fine-tuning strategy** on both the **S** and **s10** datasets. During training, layers are gradually unfrozen from top to bottom while reducing the learning rate at each stage.

## 🔁 3-Stage Training Procedure

1. **🔹 Stage 1: Fine-tune Fully Connected Layer**
   - **Learning rate**: `1e-3`
   - **Weight decay**: `1e-4`
   - **Layers trained**: Only the final fully connected (FC) layer
   - **Epochs**: 3

2. **🔸 Stage 2: Fine-tune Top Layers**
   - **Learning rate**: `1e-4`
   - **Weight decay**: `1e-4`
   - **Layers trained**: Last 3 convolutional layers (PyTorch: `layer4[2]`) and FC layer
   - **Epochs**: 10

3. **🔻 Stage 3: Fine-tune Entire Network**
   - **Learning rate**: `1e-5`
   - **Weight decay**: As previously set
   - **Layers trained**: All layers in ResNet50
   - **Epochs**: 37

During training, we augment mammograms to promote model generalizability by applying the following augmentations:
- Horizontal and vertical flips 
- Rotations in [-25,25] degrees
- Zoom in [0.8,1.2] ratio
- Intensity shift in [-20,20] % of pixel values
- Shear in [-12,12] grades

We train the Resnet50 for 50 epochs in total. However, since the S dataset is much smaller than s10, we increase the number of epochs in the third stage to 100. The batch size is 256, and we use ADAM as the optimizer.  The model's parameters are initialized with the pre-trained weights in Imagenet.

##### Train patch classifier by using train_function.py in "patch_classifier" folder 
##### Trained Patch-level classifier models are in trained_models 

| Dataset      | Validation acc.| Test acc.     |
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

**Note**: The backbone used in the image classifier corresponds to the ResNet50 trained on the s10 patch dataset. 

##### Trained whole image classifier model is  in the "trained_models" folder 

| Model                    | Test       Acc.| Test AUC.     |
| :---                     |     :---:      |          ---: |
| ResNet50+2 ResNet Blocks |     0.857      | 0.856         |
                                

### Heatmaps

In the paper, the trained patch classifier was utilized in a sliding window manner across the entire image to generate a heatmap indicating the location of the lesions. This can be imagined as a convolutional operation over an image, where instead of performing the dot product between the receptive field and the filter, the receptive field is input into the patch classifier to obtain a value ranging from 0 to 1. The size of the heatmap depends on the dimensions of the mammograms and the patch, as well as the stride at which the patch classifier is moved across the image and the padding.![Example of three different heatmaps](https://github.com/sposso/Deep_learning_to_improve_breast_Cancer_pytorch/blob/main/Heatmaps/heatmaps.png)

##### Generate heatmaps by using the function my_prob_heatmap contained in /Heatmaps/main.py





