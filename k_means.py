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
from sklearn import preprocessing
import EVALUATION





### -------------------------------------------- K-means clustering algorithm -------------------------------------

# TODO QUESTION: Do I initialize correctly the points? (Arbitrary in the domain or choose one of the sample points?) --> Question on Discord.
# TODO TASK: Decimal accuracy. I don't take all the decimals as in the txt. Impacts my convergence value selection.

def k_means(df_nl, df_feat):
    # TODO: POSSIBLY I HAVE TO CUT THE DATASET IN THE ALGORITHM
    # TODO: SO I SHOULD PROVIDE THE WHOLE DATASET IN THE FUNCTION, AND THE FUNCTION CUTS THE DATASET/


    df_final = df_nl # εδώ θα προστεθεί το αποτέλεσμα του cluster
    df = df_feat

    # TODO: NORMALIZE THE DATAFRAME
    # Normalization = each attribute value / max possible value of this attribute
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df)
    df = pd.DataFrame(x_scaled)
    df.columns = ["density_2d", "proj_area", "height"]
    # df.columns = ["volume", "proj_area", "area_3d", "height", "density_2d","density_3d"]
    # print("df_normalized: \n", df)


    global clusters_objects, clusters
    k = 5
    lists = [] # list that holds all the (sub)lists of separate features, e.g. lists = [[feat_1],[feat_2],[feat_3]]
    centroids = [] # list that holds the feature vector of the centroid !

    # STEP 1: initialize the centroids
    # TODO TASK: IT CAN BE SIMPLIFIED !!!
    # Find my domain and create a resolution
    cols = int(len(df.columns)) # TODO: possibly it's gonna be different when I will have my features
    for c in df.columns: # where c is the x,y,z --> the columns of the dataframe
        list_xyz = df[c].tolist() # add all the column x in a sublist in the 'lists' list, and so on for the rest of the columns
        lists.append(list_xyz) # append the sublist to the list 'lists'
    mini = min(lists[0]) # ask for the smallest value of the first feature 'x'
    maxi = max(lists[0]) # and for the biggest value
    res = (maxi - mini) / k  # find the resolution

    # Initialize my k centroids. I force the centroids to initialize far away from each other, forcing the first feature to be
    # taking values in a grid format. e.g. the first centroid is initialized in min_x and the second centroid is initialized in
    # min_x + resolution of my domain for with as many cells as the number of the clusters.
    for d in range(k):
        centroid = []
        for i in range(cols):
            if i == 0:
                centroid.append(mini)
                mini = mini + res
            else:
                centroid.append(random.choice(lists[i]))
        centroids.append(centroid)

    # Work with numpy arrays --> the k centroids in an array
    centroids_arr = np.array(centroids)
    # print(centroids_arr)

    # turn the dataframe holding the points in an array
    points_arr = df.to_numpy()
    # print(points_arr)

    # Start the while loop
    n_iter = 0
    while n_iter < 100:

        # 5 lists for our 5 clusters --> will hold the index of the objects  belonging to them
        # in every iteration is empty
        cl_0 = []
        cl_1 = []
        cl_2 = []
        cl_3 = []
        cl_4 = []

        # STEP 2: assign each object to a cluster

        for i in range(len(points_arr)):
            dists_list = [] # save the k distances to the k clusters to this list
            for j in range(len(centroids_arr)): # traverse each of the k centroid vectors
                distance = 0
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
                df_final.loc[i, 'cluster'] = 0
            if min_index == 1:
                cl_1.append(i)
                df_final.loc[i, 'cluster'] = 1
            if min_index == 2:
                cl_2.append(i)
                df_final.loc[i, 'cluster'] = 2
            if min_index == 3:
                cl_3.append(i)
                df_final.loc[i, 'cluster'] = 3
            if min_index == 4:
                cl_4.append(i)
                df_final.loc[i, 'cluster'] = 4

        clusters_objects = [] # a list of dataframes
        # save the objects of the same cluster as one dataframe in the list of dataframes called clusters_objects
        clusters_objects.append(df.iloc[cl_0]) # all the rows
        clusters_objects.append(df.iloc[cl_1])
        clusters_objects.append(df.iloc[cl_2])
        clusters_objects.append(df.iloc[cl_3])
        clusters_objects.append(df.iloc[cl_4])

        clusters = []
        clusters.append(cl_0)
        clusters.append(cl_1)
        clusters.append(cl_2)
        clusters.append(cl_3)
        clusters.append(cl_4)

        # STEP 3: Calculate the current centroids and update the value of centroids_arr
        # save the previous centroids in a variable for comparison
        prev_centroids_arr = centroids_arr
        cur_centroids_arr = []
        dists = []
        l = 0
        for n in clusters_objects:
            if len(n) > 1:
                # within every cluster, calculate the mean centroid and append to the cur_centroids_arr
                # print("The DataFrame with mean values of each column is:")
                cur_centroid = n.mean(axis=0).to_numpy()
                cur_centroids_arr.append(cur_centroid)
                dist = np.sqrt(np.sum(np.square(prev_centroids_arr[l] - cur_centroid)))
                dists.append(dist)
                l += 1

        # check if all centroids converged
        if all(d < 0.00000001 for d in dists):
            print("Convergence achieved in iteration", n_iter)
            break
        # update the centroids
        centroids_arr = cur_centroids_arr

        n_iter += 1  # increase the iteration number every time with step 1



    clusters_list = [i for i in range(len(df))]
    cluster_index = 0
    for li in clusters:
        for p_index in li:
            clusters_list[p_index] = cluster_index
        cluster_index += 1


    print("df_final: \n", df_final)

    # df3 = df.combine_first(df_final)
    # print(df3)

    # TODO: EVALUATION
    error_matrix = EVALUATION.create_error_matrix(df_final, k)

    # PLOT
    X = np.array(df_feat)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=clusters_list, cmap='Set1')
    plt.show()

    return df_final, k






