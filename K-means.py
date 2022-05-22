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

    # STEP 1:
    # Initialise k clusters' centroids (c1, c2, …, ck) in a way such that  the initial centroids are placed as far as possible from each other.
    cols = int(len(df.columns))
    # print("cols:", cols)
    for c in df.columns:
        list_xyz = df[c].tolist()
        lists.append(list_xyz)

    # create the resolution based on the first feature
    mini = min(lists[0])
    maxi = max(lists[0])
    res = (maxi - mini) / k
    # print("mini, maxi, res:", mini, maxi, res)

    # initialize the k centroids far from each other
    for k in range(k):
        centroid = []
        for i in range(cols):
            if i == 0:
                centroid.append(mini)
                mini = mini + res
            else:
                centroid.append(random.choice(lists[i]))
        # add the centroid to the list of centroids --> k in number. As many as the number of classes.
        centroids.append(centroid)

    # turn the centroids into numpy element
    centroids_arr = np.array(centroids)
    # print(centroids) # list
    # print(centroids_arr) # array

    # # PLOTTING ok.
    # ax = plt.axes(projection='3d')
    # for i in centroids:
    #     ax.plot3D(i[0], i[1], i[2], 'gray')
    #     ax.scatter3D(i[0], i[1], i[2], cmap='Greens')
    # plt.show()

    # print("len(df.index): ",len(df.index))
    # print(df.index) # start=0, stop=..., step=1

    # add all the rows of dataframe to an array where every row of the dataframe is a row in the array
    points_arr = df.to_numpy()
    # # check if the 500 objects are there
    # print(len(points_arr))
    # print("type:",type(points_arr))

    n_iter = 0
    while n_iter < 60:
        print("n_iter: ", n_iter)
        n_iter +=1
        # Initialize one list for every cluster
        cl_0 = []
        cl_1 = []
        cl_2 = []
        cl_3 = []
        cl_4 = []

        # STEP 2: assign to the object ... the index of the closest centroid
        distance = 0
        for i in range(len(points_arr)): # fow all the rows of the dataframe
            dists_list = []
            for j in range(len(centroids_arr)):
                for c in range(cols): # it will gradually add all the squares of the cols
                    distance = distance + (points_arr[i][c] - centroids_arr[j][c])**2
                # when the loop of cols finishes, it's time to take the sqrt of the distance
                # because it calculated the distances between all the corresponding features vectors
                distance = math.sqrt(distance)
                ## add this distance to the list of distances --> there are gonna be as many distances as the number of clusters j
                dists_list.append(distance)
            # then, we return to calculate the distance of the first object to the next cluster-centroid
            # if the next distance calculation is easier, we take the bigger
            # or alternatively we add the k distances to a list and we retrieve the index of the bigger element
            min_value = min(dists_list)
            min_index = dists_list.index(min_value)
            # print("dists_list: ", dists_list)
            # print("min_index: ", min_index)

            # take the object's index (i) and save it into a list that holds similar objects
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

        # printing-statements to check if the objects are correctly saved into the list of clusters
        # print(cl_0)
        # print(df.iloc[cl_0])

        # use these lists that hold each cluster to retrieve the correct rows from the dataframe
        clusters_objects = [] # a list of dataframes
        # add to the above lists the dataframes
        clusters_objects.append(df.iloc[cl_0])
        clusters_objects.append(df.iloc[cl_1])
        clusters_objects.append(df.iloc[cl_2])
        clusters_objects.append(df.iloc[cl_3])
        clusters_objects.append(df.iloc[cl_4])

        # STEP 3: Calculate the current centroids
        # save the previous centroid
        prev_centroids_arr = centroids_arr
        cur_centroids_arr = []
        l = 0 # iterator to traverse the previous_centroids_array
        for n in clusters_objects: # within every cluster
            print("l: ", l)
            # Average of each column using DataFrame.mean()
            cur_centroid_arr = n.mean(axis=0).to_numpy()
            # save the new centroid to an array which holds all the centroids
            cur_centroids_arr.append((cur_centroid_arr))
            dist = np.sqrt(np.sum(np.square(prev_centroids_arr[l] - cur_centroid_arr)))
            print("prev_centroids_arr[l]: ", prev_centroids_arr[l])
            print("cur_centroids_arr: ", cur_centroids_arr)
            print("dist: ", dist)
            print("")
            # after every iteration, increase the iterator l by one step
            l+=1
            # Now check the difference between the previous and the current centroids
            # if dist < 0.00000001:
            #     break

        # save the current centroids array to the variable centroids_arr and reinitialize the while loop
        centroids_arr = cur_centroids_arr





















# call k_means function
k_means(df)




























