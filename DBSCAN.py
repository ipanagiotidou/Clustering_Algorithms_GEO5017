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
from sklearn.decomposition import PCA


### -------------------------------------------- DBSCAN clustering algorithm -------------------------------------
# TODO: DBSCAN algorithm
# TODO: chose the epsilon (ε) parameter
# TODO: Choose the minimum points (MinPts) parameter
# TODO: BE VERY CAREFUL: THE KDTREE RETURNS ITSELFT AS WELL.

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

def dbscan(dt, minPts, epsilon):
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

        # increase the cluster index before visiting next point of the dataset
        cl_id += 1

    print(cl_of_pts)
    # return the list which holds in which cluster each point belongs to
    return cl_of_pts


