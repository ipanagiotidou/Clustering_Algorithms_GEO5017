import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
import validation_method


### -------------------------------------------- K-means clustering algorithm -------------------------------------
# TODO TASK: Decimal accuracy. I don't take all the decimals as in the txt. Impacts my convergence value selection.

def k_means(df_nl, df): # where df_nl is the dataframe with basename and label and df the dataframe with features only

    # TODO: NORMALIZE the features = each attribute value / max possible value of this attribute
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df)
    df = pd.DataFrame(x_scaled)
    # TODO: add here the df.columns names --> ...
    # print("df_normalized: \n", df)


    global clusters_objects, clusters
    k = 5
    lists = []
    centroids = []

    # STEP 1: initialize the centroids
    # Find my domain and create a resolution metre based on the number of clusters and range of first feature.
    cols = int(len(df.columns))
    for c in df.columns:
        list_xyz = df[c].tolist() # add the first feature values of all the objects in a list
        lists.append(list_xyz) # then append the above sublist to the 'lists'
    mini = min(lists[0]) # ask for the smallest, and biggest values of the first feature 'x'
    maxi = max(lists[0])
    res = (maxi - mini) / k  # find the resolution

    # Forcing the centroids to initialize far away from each other:
    # a) Force the first feature of every centroid to take values min(x) and then min(x)+resolution and so on.
    # b) Enforce randomness with letting the other features take random values from the dataset.
    for d in range(k):
        centroid = []
        for i in range(cols):
            if i == 0:
                centroid.append(mini)
                mini = mini + res
            else:
                centroid.append(random.choice(lists[i]))
        centroids.append(centroid)

    # Centroids in array format
    centroids_arr = np.array(centroids)

    # Objects in array format
    points_arr = df.to_numpy()

    # Start the loop of refinement of the centroids.
    n_iter = 0
    while n_iter < 100:

        # 5 lists for the 5 clusters
        cl_0 = []
        cl_1 = []
        cl_2 = []
        cl_3 = []
        cl_4 = []

        # STEP 2: assign each object to a cluster
        # measure the distance between every object and each of the k centroids

        for i in range(len(points_arr)): # traverse every object
            dists_list = []
            for j in range(len(centroids_arr)): # traverse each of the k centroid vectors
                distance = 0
                for c in range(cols):
                    distance = distance + (points_arr[i][c] - centroids_arr[j][c])**2
                distance = math.sqrt(distance)
                # add the calculated distance to the dists_list.
                dists_list.append(distance)
            # find the index of the minimun distance to find the closest centroid
            min_value = min(dists_list)
            min_index = dists_list.index(min_value)

            # based on the returned index (min_index), I assign the object to a cluster.
            if min_index == 0:
                cl_0.append(i)
                df_nl.loc[i, 'cluster'] = 0
            if min_index == 1:
                cl_1.append(i)
                df_nl.loc[i, 'cluster'] = 1
            if min_index == 2:
                cl_2.append(i)
                df_nl.loc[i, 'cluster'] = 2
            if min_index == 3:
                cl_3.append(i)
                df_nl.loc[i, 'cluster'] = 3
            if min_index == 4:
                cl_4.append(i)
                df_nl.loc[i, 'cluster'] = 4

        clusters_objects = [] # a list of dataframes
        # with df.iloc[cl_0] I locate all the rows from the initial dataframe that belong to cluster 0 and so on.
        clusters_objects.append(df.iloc[cl_0]) # cl_0 is a list, I pass a list with indices to iloc function.
        clusters_objects.append(df.iloc[cl_1])
        clusters_objects.append(df.iloc[cl_2])
        clusters_objects.append(df.iloc[cl_3])
        clusters_objects.append(df.iloc[cl_4])

        # Helpful for Plotting --> create a list with sublists, one for every cluster
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


    # Helpful for Plotting --> return a list with indices, indicating the cluster in which every object belongs to.
    # Pass cluster_list to Plotting. c=cluster_list
    clusters_list = [i for i in range(len(df))]
    cluster_index = 0
    for li in clusters:
        for p_index in li:
            clusters_list[p_index] = cluster_index
        cluster_index += 1


    # print the clustered version of the dataframe of objects
    # print("df_nl: \n", df_nl)

    # TODO: EVALUATION method
    error_matrix = validation_method.create_error_matrix(df_nl, k)

    # TODO: PLOT
    X = np.array(df)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=clusters_list, cmap='Set1')
    plt.show()

    # todo: return
    return df_nl, k






