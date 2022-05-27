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
import Load_the_data

# check the algorithms
from sklearn import datasets
from sklearn.decomposition import PCA

### -------------------------------------------- HIERARCHICAL clustering algorithm -------------------------------------
# TODO: Hierarchical algorithm
# TODO: read the comments. delete unnecessary


# TODO: MAKE SURE THIS VALUE WORKS FOR YOUR DATASET  --> --> --> --> Change 10**2 value
n = 10**2

def compute_matrix(my_data):
    # turn the data into an array
    my_data = np.array(my_data)
    l = len(my_data) # save the length of my_data

    # initialize an empty 2D matrix to hold the distances between the objects in the size of our data (rows and columns)
    matrix = np.zeros(shape=(l,l))

    # fill the matrix
    for i in range(l):
        for j in range(l):

            if i!=j: # calculate distances only between different objects (not in the diagonal)
                # TODO: to be deleted if not used --> MANHATAN distance
                # Manhatan distance --> Replace with Euclidean
                # matrix[i][j] = sum(abs(val1-val2) for val1, val2 in zip(my_data[i],my_data[j]))
                # matrix[j][i] = matrix[i][j]
                # TODO: delete the above if not used.

                # We calculate the Euclidean distance between the feature vectors of the objects
                val1 = my_data[i]
                val2 = my_data[j]
                matrix[i][j] = round(np.linalg.norm(val1-val2), 2)

            else:
                matrix[i][j] = n   # assign a big value in the diagonal elements so that we never get them back as the minimum value

    return matrix


def new_cluster(matrix):
    '''
    The function takes a matrix and returns the row and column (indices) where it found the minimum value.
    These indices indicate the objects that should be grouped in the same cluster.
    Returns the indices in ascending order, in string format since I will use them as keys to retrieve values from a dictionary.
    '''
    c = []

    indices = np.where(matrix == np.amin(matrix))
    c.append(str(indices[0][0]))
    c.append(str(indices[1][0]))

    if (c[0] < c[1]):
        return str(int(c[0])), str(int(c[1]))
    else:
        return str(int(c[1])), str(int(c[0]))


def reformat_matrix(matrix, key1, key2):
    '''
    The function takes as parameters the matrix and two indices.
    These two indices correspond to the objects that should be grouped into the same cluster.

    We reformat the Matrix of distances keeping the smallest distance between the objects and the new cluster.
    So, for example, if we have to group objects 2 and 5, we will reassign distance values to all the rest of the objects as:
    d(0,2) = min ( d(0,2), d(0,5) ).
    Finally, in the row and column of the matrix belonging to object 5 we will assign a very large value
     so that it will never be given back as the smallest distance. Meaning...we will never be forced to group it to a cluster!
    '''

    # turn the keys of the dictionary from string to integers to use them for indexing.
    key1 = int(key1)
    key2 = int(key2)
    for i in range(matrix.shape[0]): # loop by using the row dimension of the matrix
        if i!=key1 or i!=key2: # don't do the below in the rows and cols of the objects that are grouped together
            matrix[i][key1] = min(matrix[i][key1], matrix[i][key2])
            matrix[key1][i] = matrix[i][key1] # do the opposite because the matrix is symmetric

        # assign a very big number to the rows and cols of the second object (key2)
        matrix[i][key2] = n
        matrix[key2][i] = n
    # fill the diagonal again with big values
    np.fill_diagonal(matrix, n)

    # print("matrix_new: \n ", matrix)
    return matrix


def hierarchical_nail(df, number_of_clusters):

    # assign the desired number of clusters into variable m
    m = number_of_clusters

    # assign the dataset to variable X
    X = df

    # MATRIX
    # call function that computes the initial matrix --> distances between all objects
    matrix = compute_matrix(X)

    # CLUSTERS:
    # Initialize a dictionary with one key-value pair for every object
    clusters = {}                  # e.g. {'0': [0], '1': [1], '2': [2], '3': [3]}
    for i in range(len(X)):
        clusters[str(i)]=[i]
    # print(clusters)

    # KEYS:
    # call the function that returns the indices of row and column with the minimum value(distance) in the matrix
    key1, key2 = new_cluster(matrix)
    # print("keys: ", key1, key2)

    clusters[key1] = clusters[key1] + clusters[key2]
    del clusters[key2]

    l = len(clusters)

    matrix_new = reformat_matrix(matrix, key1, key2)

    while l > m:
        key1, key2 = new_cluster(matrix_new)
        # print("keys: ", key1, key2)

        clusters[key1] = clusters[key1] + clusters[key2]
        del clusters[key2]

        # print clusters
        # print("clusters: ", clusters)

        # update the matrix
        matrix_new = reformat_matrix(matrix_new, key1, key2)

        # remove one cluster from the list
        l = l - 1


    # TODO TASK: save in the initial dataframe the label of the clusters that each object belongs to
    # for every key in the dictionary holding the clusters
    for key in clusters.keys():
        # use the key and retrieve the values
        # since every key is a cluster --> the values will be the objects that belong to this cluster
        # locate the row where the object is based on its index (val) and assign to the column 'cluster' its dictionary key
        for val in clusters[key]:
            df.loc[val, 'cluster'] = int(str(key))

    # print("clusters: \n", clusters)
    # print("df: \n", df)


    # return df # alternative return
    return clusters






