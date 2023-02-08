from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os 
import torchvision.transforms as T
#from ..utils.tools import CBIS_MAMMOGRAM
#Current working directory 



df = pd.read_csv('/home/sposso22/Desktop/first_project/data/sample.csv')
target = df.labels.to_numpy()
train_indices, test_indices = train_test_split(np.arange(target.shape[0]), test_size= 0.15, train_size=0.85, stratify=target, random_state= 42)
df_train = df.loc[train_indices,:]
df_test = df.loc[test_indices, :]
df_test = df_test.reset_index(drop = True)
df_test.to_csv('/home/sposso22/Desktop/first_projec/data/test.csv', index = False)
df_train = df_train.reset_index(drop = True )
label_train = df_train.label.to_numpy()
train_in, validation_in = train_test_split(np.arange(train_indices.shape[0]), test_size =0.1, train_size = 0.9, stratify =label_train, random_state= 42)
d_train = df_train.loc[train_in, :]
d_train = d_train.reset_index(drop = True)
d_train.to_csv('/home/sposso22/Desktop/first_projec/data/test.csv', index = False)
d_validation = df_train.loc[validation_in,:]
d_validation = d_validation.reset_index(drop = True)
d_validation.to_csv('/home/sposso22/Desktop/first_projec/data/validation.csv',index = False)


    








    





