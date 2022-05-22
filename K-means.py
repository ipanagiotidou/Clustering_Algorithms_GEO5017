import numpy as np
import pandas as pd
import random
import math
import scipy.spatial
from scipy.spatial import ConvexHull, distance_matrix
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
import csv

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
        df_xyz = pd.read_table(path + file, delimiter=' ')
        df_xyz.columns = ["x", "y", "z"]
        # 1) assign the label in the label col, 2) save the base_name as integer in the bname col
        df_xyz.loc[:,'bname'] = iname
        df_xyz.loc[:,'label'] = label

        # merge all sub-dataframes to the initialized dataframe
        df = pd.concat([df, df_xyz], sort=False, ignore_index=True)

# print the dataframe and columns=index
# print(df)
# print(df.columns)

### - - - - - - - - - - - - - - - - - - - - K-means clustering algorithm - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def k_means(features_df):
    k = 5
# ---- to be deleted - start ----
    selected_columns = features_df[["x", "y", "z"]]
    df = selected_columns.copy()
    df = df.astype(float)
    # print(df)
# ---- to be deleted - end ----

    lists = [] # list that holds all the (sub)lists of seperate features, e.g. lists = [[feat_1],[feat_2],[feat_3]]
    centroids = [] # list that holds the feature vector of the centroid !

    # STEP 1: initialize the centroids
    cols = int(len(df.columns))
    for c in df.columns:
        list_xyz = df[c].tolist()
        lists.append(list_xyz)
    mini = min(lists[0])
    maxi = max(lists[0])
    res = (maxi - mini) / k

    for d in range(k):
        centroid = []
        for i in range(cols):
            if i == 0:
                centroid.append(mini)
                mini = mini + res
            else:
                centroid.append(random.choice(lists[i]))
        centroids.append(centroid)

    # Work with numpy arrays
    centroids_arr = np.array(centroids)
    points_arr = df.to_numpy()

    # Start the while loop
    n_iter = 0
    while n_iter < 100:
        n_iter +=1

        # lists that hold the number of the objects that belong to them
        cl_0 = []
        cl_1 = []
        cl_2 = []
        cl_3 = []
        cl_4 = []

        # STEP 2: assign each object to a cluster
        distance = 0
        for i in range(len(points_arr)):
            dists_list = [] # save the k distances to the k clusters to this list
            for j in range(len(centroids_arr)): # traverse each of the k centroid vectors
                for c in range(cols): # calculate the distance between the corresponding cols
                    distance = distance + (points_arr[i][c] - centroids_arr[j][c])**2
                distance = math.sqrt(distance)
                # add the calculated distance to the dists_list.
                dists_list.append(distance)
            # find the index of the minimun distance
            min_value = min(dists_list)
            min_index = dists_list.index(min_value)

            # add the objects to the correct list based on the returned index (min_index)
            if min_index == 0:
                cl_0.append(i)
            if min_index == 1:
                cl_1.append(i)
            if min_index == 2:
                cl_2.append(i)
            if min_index == 3:
                cl_3.append(i)
            if min_index == 4:
                cl_4.append(i)

        clusters_objects = [] # a list of dataframes
        # save the objects of the same cluster as one dataframe in the list of dataframes called clusters_objects
        clusters_objects.append(df.iloc[cl_0])
        clusters_objects.append(df.iloc[cl_1])
        clusters_objects.append(df.iloc[cl_2])
        clusters_objects.append(df.iloc[cl_3])
        clusters_objects.append(df.iloc[cl_4])

        # STEP 3: Calculate the current centroids and update the value of centroids_arr
        # save the previous centroids in a variable for comparison
        prev_centroids_arr = centroids_arr
        cur_centroids_arr = []
        dists = []
        l = 0
        for n in clusters_objects:
            # within every cluster, calculate the mean centroid and append to the cur_centroids_arr
            cur_centroid_arr = n.mean(axis=0).to_numpy()
            cur_centroids_arr.append((cur_centroid_arr))
            dist = np.sqrt(np.sum(np.square(prev_centroids_arr[l] - cur_centroid_arr)))
            dists.append(dist)
            l += 1
        # check if all distances are
        if all(d < 0.00000001 for d in dists):
            print("Convergence achieved in iteration", n_iter)
            break
        # update the centroids
        centroids_arr = cur_centroids_arr

    # after exiting the while loop
    return(clusters_objects)


# - - - - - - - - - - - - - - - - - - Call the clustering_algorithms - - - - - - - - - - - - - - - - - - - - -
# - - - - - - call K-means - - - - - -
k_means(df)