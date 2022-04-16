import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from IGTD import table_to_image
import os

#takes pandas dataframe as input and returns min-max normalized dataframe for the same
def min_max_normalizer(data):
    data_copy=data.copy()
    for col in data_copy.columns:
        m=data_copy[col].min()
        M=data_copy[col].max()
        if type(data_copy[col][0])!=np.bool_:
            data_copy[col]=(data_copy[col]-m)/(M-m)
    data =data_copy
    return data
    
#implementing Smote oversampling procedure
def smote_transform(data):
    output=data.columns[-1]
    X=data.drop([output], axis=1)
    y=data[output]
    column_names=pd.concat([X,y], axis=1).columns.tolist()
    X=data.drop([output], axis=1).to_numpy()
    y=data[output].to_numpy()
    smt=SMOTE()
    X,y=smt.fit_resample(X,y)
    smote_arr=np.concatenate([X,y.reshape(-1,1)], axis=1)
    data=pd.DataFrame(smote_arr, columns=column_names)
    return data

#Transforming tabular data into image data
def pre_process(data_set):
    data_set=min_max_normalizer(data_set)
    data_set=smote_transform(data_set)
    output=data_set.columns[-1]
    only_features=data_set.drop([output], axis=1)
    only_results=data_set[output]
    num_of_features=only_features.columns.size
    num_rows=1
    num_columns=num_of_features
    for i in range(2,num_of_features):
        if num_of_features % i == 0 :
            num_rows=i
            num_columns=num_of_features//i
            break
    feature_distance_method="Euclidean"
    pixel_distance_method="Euclidean"
    save_image_size=3.0
    max_step=10000
    val_step=300
    error="squared"
    result_dir='Results/Test_1'
    os.makedirs(name=result_dir, exist_ok=True)
    table_to_image(only_features, [num_rows, num_columns], feature_distance_method, pixel_distance_method, save_image_size, max_step, val_step, result_dir, error)
    return data_set