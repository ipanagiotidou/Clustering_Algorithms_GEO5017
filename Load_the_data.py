
### -------------------------------------- Libraries ----------------------------------------

import pandas as pd
import os


### ----------------------------------------- Load the data -----------------------------------------
def load_the_data():
    # TODO TASK: Don't forget to change the path ! --> ''...objects/data''
    files = os.listdir("C:/Users/Panagiotior/Desktop/Q3/MachineLearning/hw/GEO5017-A1-Clustering/scene_objects/scene_objects/data2")
    path = "C:/Users/Panagiotior/Desktop/Q3/MachineLearning/hw/GEO5017-A1-Clustering/scene_objects/scene_objects/data2/"

    # create a dataframe to hold all sub-dataframes created in the loop below
    # where 'bname' stands for the base_name of the file, and 'label' for the clustering category
    df = pd.DataFrame(columns=['x','y','z','bname','label']) # set the column names

    # do through the files and save each file in a dataframe called 'df_xyz'.
    # at the end of the loop add (concatenate) this dataframe to the big one called 'df'
    for file in files:
        file_extension = os.path.splitext(file)[-1].lower() # search for files ending in .xyz
        base_name = os.path.splitext(file)[-2].lower() # reads the 3-digits integer base name, e.g. 499
        iname = int(base_name) # convert the base_name into an integer # TODO: make sure it works for the first base names, e.g. 001, 002, ...
        if file_extension == ".xyz":
            # determine the label name based on the base_name of the file, e.g. from 0 (integer of 000) to 099 it is a building.
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

            # read the .TABLE(.xyz file). and assign x,y,z to the corresponding columns
            # TODO TASK: when I will calculate the features I have to change this part. Either only the names of the 3 features or add more.
            df_xyz = pd.read_table(path + file, delimiter=' ', header=None) # read the .xyz file with a delimiter 'space'
            df_xyz.columns = ["x", "y", "z"] # set the column names of my small dataframe 'df_xyz'

            # print(df_xyz) # so far I only have a dataframe with columns x,y,z.

            # 1) assign the label in the 'label' column
            # 2) save the base_name as integer in the 'bname' column
            # Tool: using the 'loc' function I can access a group of rows and columns by label(s).
            # Here I choose all the rows and column 'bname' and assign the value of variable iname, and label, respectively.
            df_xyz.loc[:,'bname'] = iname
            df_xyz.loc[:,'label'] = label

            # print(df_xyz) # here, I have created the whole dataset with columns x,y,z,label,bname filled with values

            # merge the sub-dataframe 'df_xyz' created for the specific file to the initialized dataframe 'df' that holds all the files
            df = pd.concat([df, df_xyz], sort=False, ignore_index=True)

    # print the dataframe, and columns
    # print(df)
    # print(df.columns)
    return df

def load_half_the_data(df):
    df = load_the_data()
    dt = df[["x", "y", "z"]].copy().astype(float).to_numpy()  # it is an array
    # print(dt)
    return dt