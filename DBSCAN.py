import numpy as np
import pandas as pd
import scipy.spatial
from sklearn import preprocessing
import matplotlib.pyplot as plt
import validation_method


### -------------------------------------------- DBSCAN clustering algorithm -------------------------------------
# TODO: DBSCAN algorithm
# TODO SOS: chose the epsilon (ε) parameter
# TODO SOS: Choose the minimum points (MinPts) parameter


def n_indices(pt, kdtree, epsilon):
    ns_indices = kdtree.query_ball_point(pt, r=epsilon, p=2.0, workers=1, return_sorted=None, return_length=False)
    return ns_indices

def visit_ns(ns_indices, dt, cl_id, cl_of_pts, kdtree, epsilon, minPts):
    # visit the neighbours one by one
    for n_index in ns_indices:
        # if the neighbour doesn't belong to a cluster, add it to the current cluster_id.
        if cl_of_pts[n_index] == -1:
            cl_of_pts[n_index] = cl_id
            # If the current neighbour is a CORE point then call again the same function
            new_n_indices = n_indices(dt[n_index], kdtree, epsilon)
            if (len(new_n_indices)) >= minPts: # If core point test
                visit_ns(new_n_indices,  dt, cl_id, cl_of_pts, kdtree, epsilon, minPts)

def dbscan(df_nl, df_feat, minPts, epsilon):

    # TODO: NORMALIZE THE DATAFRAME = each attribute value / max possible value of this attribute
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df_feat)
    df = pd.DataFrame(x_scaled)
    print("df_normalized: \n", df)

    # turn the features dataframe into an array
    dt = df_feat.to_numpy()

    # create the kdtree of all the points
    kdtree = scipy.spatial.KDTree(dt)

    # initiate a cluster id
    cl_id = 0
    # cl_of_pts is a list containing the cluster id of every object
    cl_of_pts = [-1] * len(dt)   # in the beginning, all objects set to -1 --> meaning no cluster is assigned

    # traverse all the points
    for index, pt in enumerate(dt): # create indexing with 'enumerate' function

        # If the point belongs to a cluster, visit the next point
        if cl_of_pts[index] != -1:
            continue

        # If the point is a CORE point, that doesn't belong to a cluster, assign it to the current cluster_index (cluster id)
        n_indices_pt = n_indices(pt, kdtree, epsilon)
        if len(n_indices_pt) >= minPts: # check for core points
            cl_of_pts[index] = cl_id # update its id
            # visit its neighbours and assign to them the same cluster_index
            visit_ns(n_indices_pt, dt, cl_id, cl_of_pts, kdtree, epsilon, minPts)
            cl_id +=1


    # print(cl_of_pts)

    # add the cluster category to the datafame --> pass the row of the dataframe to the iloc function, update 'cluster' column
    for i in range(len(cl_of_pts)):
        # i is the indexing for the row
        df_nl.loc[i,'cluster'] = cl_of_pts[i]

    # todo: Count occurrence of element '-1' in numpy array
    # cl_of_pts_arr = np.array(cl_of_pts)
    # count = (cl_of_pts_arr == -1).sum()
    # print("count unclustered points: ", count)

    # TODO: call EVALUATION method
    n = len(set(cl_of_pts)) # take all unique values from the list 'cl_of_pts' to find the number of clusters
    error_matrix = validation_method.create_error_matrix(df_nl, n)

    # TODO: PLOT
    X = np.array(df)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=cl_of_pts, cmap='Set1')
    plt.show()


    # return the list which holds in which cluster each point belongs to
    return df_nl





