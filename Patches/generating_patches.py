from patches_utils import sample_patches
import pandas as pd
import os
import numpy as np
from imageio import imread



# Data were training and testing mammograms are stored.

home_data = "/home/sposso22/work/official_split"

# Path to the folder where patches will be saved.
patches_folder = os.path.join(home_data, "patches")

# Create the patches folder if it does not exist.
os.makedirs(patches_folder, exist_ok=True)

# Setting to choose whether to generate patches for training or testing data.
setting = "training"  # or "testing"


if setting == "training":
    
    training_mammograms = pd.read_csv(os.path.join(home_data, "training_loc.csv"))
    
    # Create the training patches folder.
    training_patch_folder = os.path.join(patches_folder, "training")
    os.makedirs(training_patch_folder, exist_ok=True)

    # Create the S10 patches folder.
    s10_patch_folder = os.path.join(training_patch_folder, "s10")
    os.makedirs(s10_patch_folder, exist_ok=True)
    
    # Create the S patches folder.
    s_patch_folder = os.path.join(training_patch_folder, "s")
    os.makedirs(s_patch_folder, exist_ok=True)
    
    # create dataframes to store the paths and labels of the patches.
    df_s = pd.DataFrame(columns=["path", "label"])
    df_s10 = pd.DataFrame(columns=["path", "label"])
    
    # Iterate over the training mammograms.
    
    for index in range(len(training_mammograms)):

        mam = imread(training_mammograms.iloc[index]["img_path"])
        mask = imread(training_mammograms.iloc[index]["mask_path"])
        label = training_mammograms.iloc[index]["label"]
        
        # Sample patches from the mammogram.
        sample_patches(mam, index, label, mask, s_patch_folder, s10_patch_folder,
                       df_s, df_s10, patch_size=224, pos_cutoff=.9, neg_cutoff=.35,
                       nb_bkg=11, nb_abn=10, start_sample_nb=0, verbose=True)
        
    
# Save the dataframes to CSV files.
df_s.to_csv(os.path.join(training_patch_folder, "s.csv"), index=False)
df_s10.to_csv(os.path.join(training_patch_folder, "s10.csv"), index=False)


if setting == "testing":
    
    testing_mammograms = pd.read_csv(os.path.join(home_data, "test_loc.csv"))

    # Create the testing patches folder.
    testing_patch_folder = os.path.join(patches_folder, "test")
    os.makedirs(testing_patch_folder, exist_ok=True)

    # Create the S10 patches folder.
    s10_patch_folder = os.path.join(testing_patch_folder, "s10")
    os.makedirs(s10_patch_folder, exist_ok=True)
    
    # Create the S patches folder.
    s_patch_folder = os.path.join(testing_patch_folder, "s")
    os.makedirs(s_patch_folder, exist_ok=True)
    
    # create dataframes to store the paths and labels of the patches.
    df_s = pd.DataFrame(columns=["path", "label"])
    df_s10 = pd.DataFrame(columns=["path", "label"])
    
    # Iterate over the testing mammograms.
    
    for index in range(len(testing_mammograms)):

        mam = imread(testing_mammograms.iloc[index]["img_path"])
        mask = imread(testing_mammograms.iloc[index]["mask_path"])
        label = testing_mammograms.iloc[index]["label"]
        
        # Sample patches from the mammogram.
        sample_patches(mam, index, label, mask, s_patch_folder, s10_patch_folder,
                       df_s, df_s10, patch_size=224, pos_cutoff=.9, neg_cutoff=.35,
                       nb_bkg=11, nb_abn=10, start_sample_nb=0, verbose=True)
        
# Save the dataframes to CSV files.
df_s.to_csv(os.path.join(testing_patch_folder, "s.csv"), index=False)
df_s10.to_csv(os.path.join(testing_patch_folder, "s10.csv"), index=False)
# Print completion message.
print(f"Patches generated and saved in {training_patch_folder} and {testing_patch_folder}.")
