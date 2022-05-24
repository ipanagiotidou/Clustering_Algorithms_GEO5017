
### -------------------------------------- Libraries ----------------------------------------

import pandas as pd
import os


### ----------------------------------------- Load the data -----------------------------------------
def load_the_data():
    files = os.listdir("C:/Users/Panagiotior/Desktop/Q3/MachineLearning/hw/GEO5017-A1-Clustering/scene_objects/scene_objects/data2")
    path = "C:/Users/Panagiotior/Desktop/Q3/MachineLearning/hw/GEO5017-A1-Clustering/scene_objects/scene_objects/data2/"

    # create a dataframe to hold all sub-dataframes created in the loop below
    df = pd.DataFrame(columns=['x','y','z','bname','label'])
    for file in files:
        file_extension = os.path.splitext(file)[-1].lower() # .xyz
        base_name = os.path.splitext(file)[-2].lower() # reads the 3-digits integer base name, e.g. 499
        iname = int(base_name)
        if file_extension == ".xyz":
            # determine the label name based on the base_name of the file
            if (iname >= 0 and iname < 100):
                label = 'building'
            elif (iname >= 99 and iname < 200):
                label = 'car'
            elif (iname >= 200 and iname < 300):
                label = 'fence'
            elif (iname >= 300 and iname < 400):
                label = 'pole'
            elif (iname >= 400 and iname < 500):
                label = 'tree'
            else:
                label = 'FALSE'

            # read the table and assign x,y,z to the corresponding columns
            df_xyz = pd.read_table(path + file, delimiter=' ', header=None)
            df_xyz.columns = ["x", "y", "z"]
            # 1) assign the label in the label col, 2) save the base_name as integer in the bname col
            df_xyz.loc[:,'bname'] = iname
            df_xyz.loc[:,'label'] = label

            # merge the sub-dataframe created for the specific file to the initialized dataframe
            df = pd.concat([df, df_xyz], sort=False, ignore_index=True)

    # print the dataframe and columns=index
    # print(df)
    # print(df.columns)
    return df

def load_half_the_data(df):
    df = load_the_data()
    selected_columns = df[["x", "y", "z"]]
    df = selected_columns.copy()
    df = df.astype(float)
    dt = selected_columns.copy().astype(float).to_numpy()  # it is an array
    # print(dt)
    return dt