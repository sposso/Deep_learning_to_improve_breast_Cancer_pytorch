from patches_utils import sample_patches
import pandas as pd
import os
from imageio import imread


# Folder where the info of the mammograms is stored.

home_data = "/home/sposso22/work/shared_data/breast_cancer/official_split/nonuniform_sampling_2025/final_datasets"  # or "/home/sposso22/work/shared_data/breast_cancer/resized_images_test"

# Path to the folder where patches will be saved.
patches_folder = os.path.join("/home/sposso22/work/shared_data/breast_cancer", "patches")

# Create the patches folder if it does not exist.
os.makedirs(patches_folder, exist_ok=True)

# Setting to choose whether to generate patches for training or testing data.


for setting in ["training", "testing"]:
    print(f"Generating patches for {setting} data...")


    if setting == "training":

        # training_mammograms = pd.read_csv(os.path.join(home_data, "/home/sania/shared_data/breast_cancer/official_split/patches/training/s_train.csv"))
        training_mammograms = pd.read_csv(os.path.join(home_data, "training_resized_images.csv"))
        # Path to training_mammograms : ('/home/sania/shared_data/breast_cancer/nonuniform_sampling_2025/official_split/training_dataset.csv')
        

        # Create the training patches folder.
        training_patch_folder = os.path.join(patches_folder, "training")
        os.makedirs(training_patch_folder, exist_ok=True)
        
        # training_patch_folder  path : ('/Data/Disk1/cliplab/shared_data/breast_cancer/official_split/nonuniform_sampling_2025/final_datasets/training_dataset.csv)

            
        
        # Create the S10 patches folder.
        s10_patch_folder = os.path.join(training_patch_folder, "s10")
        os.makedirs(s10_patch_folder, exist_ok=True)

        
        # Create the S patches folder.
        s_patch_folder = os.path.join(training_patch_folder, "s")
        os.makedirs(s_patch_folder, exist_ok=True)
        
        # Create dataframes to store the paths and labels of the patches.
        s_list = []
        s10_list = []
        
        for index, row in training_mammograms.iterrows():
            print(f"Processing mammogram {index + 1}/{len(training_mammograms)}: {row['img_path']}")
            
            # load png images 
            mam = imread(row["img_path"])
            mask = imread(row["mask_path"])
            label = row["label"]
            
            print(f"Label: {label}")
            
            s, s10 = sample_patches(mam, index, label, mask, s_patch_folder, s10_patch_folder,
                        s_list, s10_list, patch_size=224, pos_cutoff=.9, neg_cutoff=.35,
                        nb_bkg=11, nb_abn=10, start_sample_nb=0, verbose=True)
            
            


        # Save the dataframes to CSV files.
        df_s = pd.DataFrame(s_list)
        print(f"Number of patches in s: {len(df_s)} for training data")
        df_s10 = pd.DataFrame(s10_list)
        print(f"Number of patches in s10: {len(df_s10)} for training data")
        df_s.to_csv(os.path.join(training_patch_folder, "s.csv"), index=False)
        df_s10.to_csv(os.path.join(training_patch_folder, "s10.csv"), index=False)


    if setting == "testing":
    
        s_list = []
        s10_list = []
        
        testing_mammograms = pd.read_csv(os.path.join(home_data, "test_resized_images.csv"))

        # Create the testing patches folder.
        testing_patch_folder = os.path.join(patches_folder, "test")
        os.makedirs(testing_patch_folder, exist_ok=True)

        # Create the S10 patches folder.
        s10_patch_folder = os.path.join(testing_patch_folder, "s10")
        os.makedirs(s10_patch_folder, exist_ok=True)
        
        # Create the S patches folder.
        s_patch_folder = os.path.join(testing_patch_folder, "s")
        os.makedirs(s_patch_folder, exist_ok=True)
        
        
        # Iterate over the testing mammograms.
        
        for index in range(len(testing_mammograms)):                  

            mam = imread(testing_mammograms.iloc[index]["img_path"])
            mask = imread(testing_mammograms.iloc[index]["mask_path"])
            label = testing_mammograms.iloc[index]["label"]
            
            # Sample patches from the mammogram.
            s_list, s10_list = sample_patches(mam, index, label, mask, s_patch_folder, s10_patch_folder,
                        s_list, s10_list, patch_size=224, pos_cutoff=.9, neg_cutoff=.35,
                        nb_bkg=11, nb_abn=10, start_sample_nb=0, verbose=True)
            
        # Save the dataframes to CSV files.
        df_s = pd.DataFrame(s_list)
        print(f"Number of patches in s: {len(df_s)} for testing data")
        df_s10 = pd.DataFrame(s10_list)
        print(f"Number of patches in s10: {len(df_s10)} for testing data")
        # Save the dataframes to CSV files.
        df_s.to_csv(os.path.join(testing_patch_folder, "s.csv"), index=False)
        df_s10.to_csv(os.path.join(testing_patch_folder, "s10.csv"), index=False)
        # Print completion message.
