import numpy as np
import pandas as pd
import random
import math
import scipy.spatial
from sklearn import preprocessing
from scipy.spatial import ConvexHull, distance_matrix
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
import csv
from sklearn.decomposition import PCA


### -------------------------------------------- DBSCAN clustering algorithm -------------------------------------
# TODO: DBSCAN algorithm
# TODO SOS: chose the epsilon (ε) parameter
# TODO SOS: Choose the minimum points (MinPts) parameter
# TODO: BE VERY CAREFUL: THE KDTREE RETURNS ITSELF AS WELL.

def n_indices(pt, kdtree, epsilon):
    ns_indices = kdtree.query_ball_point(pt, r=epsilon, p=2.0, workers=1, return_sorted=None, return_length=False)
    return ns_indices

def visit_ns(ns_indices, dt, cl_id, cl_of_pts, kdtree, epsilon, minPts):
    # visit the neighbours one by one
    for n_index in ns_indices:
        # if the neighbour doesn't belong to a cluster, add it to this one.
        if cl_of_pts[n_index] == -1:
            cl_of_pts[n_index] = cl_id
            # If the current neighbour is a CORE point then call again the same function
            new_n_indices = n_indices(dt[n_index], kdtree, epsilon)
            if (len(new_n_indices)) >= minPts:
                visit_ns(new_n_indices,  dt, cl_id, cl_of_pts, kdtree, epsilon, minPts)

def dbscan(df_nl, df_feat, minPts, epsilon):

    # TODO: NORMALIZE THE DATAFRAME
    # Normalization = each attribute value / max possible value of this attribute
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df_feat)
    df = pd.DataFrame(x_scaled)
    df.columns = ["volume", "proj_area", "area_3d", "height", "density_2d","density_3d"]
    # print("df_normalized: \n", df)


    # turn the df_features into a numpy array
    dt = df_feat.to_numpy()

    kdtree = scipy.spatial.KDTree(dt)
    cl_id = 0
    cl_of_pts = [-1] * len(dt)  # at this point none of the points belong to a cluster

    # traverse all the points
    for index, pt in enumerate(dt):
        # If the point belongs to a cluster
        if cl_of_pts[index] != -1:  # if the point belongs to a cluster, continue
            continue

        # If the point is a CORE point, that doesn't belong to a cluster, assign it a cluster_index
        n_indices_pt = n_indices(pt, kdtree, epsilon)
        if len(n_indices_pt) >= minPts:
            cl_of_pts[index] = cl_id # update its
            # visit the neighbs of the CORE point, and assign to them the same cluster_index
            visit_ns(n_indices_pt, dt, cl_id, cl_of_pts, kdtree, epsilon, minPts)
            cl_id +=1

        # TODO: before I had the cl_id here, but every time the previous if didn't give a cluster, the id was increased without a reason.
        # TODO: so...I moved it in the if statement
        # TODO: now, I increase the cluster id in case I update
        # increase the cluster index before visiting next point of the dataset
        # cl_id += 1

    # print(cl_of_pts)

    #     # TODO: DO WHAT YOU DID EARLIER TO SAVE TO THE CORRECT COLUMNS THE OUTPUT OF THE CLUSTERING.
    for i in range(len(cl_of_pts)):
        # i is the indexing for the row
        df_nl.loc[i,'cluster'] = cl_of_pts[i]

    df3 = df_nl.combine_first(df_feat)
    print(df3)



    # return the list which holds in which cluster each point belongs to
    return df3


