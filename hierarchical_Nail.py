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

### -------------------------------------------- HIERARCHICAL clustering algorithm -------------------------------------
# TODO: Hierarchical algorithm
# TODO: Change 10**2 value

n = 10**3

def compute_matrix(my_data):
    my_data = np.array(my_data)
    l = len(my_data)

    matrix = np.zeros(shape=(l,l))
    # fill the matrix
    for i in range(l):
        for j in range(l):
            if i!=j:
                # Manhatan distance --> Replace with Euclidean
                matrix[i][j] = sum(abs(val1-val2) for val1, val2 in zip(my_data[i],my_data[j]))
                matrix[j][i] = matrix[i][j]

                # # Euclidean distance
                # val1 = my_data[i]
                # val2 = my_data[j]
                # matrix[i][j] = round(np.linalg.norm(val1-val2), 3) # for val1, val2 in zip(my_data[i],my_data[j]))
            else:
                matrix[i][j] = n # in the diagonal
    return matrix # matrix_a


# this function computes the distances and returns the clusters in a list. [[], []]
def new_cluster(matrix):
    # function that takes a matrix and chooses which two clusters will form a new cluster
    clusters = []
    # print(matrix)
    indices = np.where(matrix == np.amin(matrix))
    #print("indices:",indices)
    clusters.append(str(indices[0][0]))
    clusters.append(str(indices[1][0]))
    # remove the two clusters from the previous cluster
    if (clusters[0] < clusters[1]):
        return clusters[0], clusters[1]
    else:
        return clusters[1], clusters[0]


def reformat_matrix(matrix, key1, key2):
    # TODO - question: can it be that key2 > key1 --> the way I return the keys in the function 'new_cluster' ?
    key1 = int(key1)
    key2 = int(key2)

    for i in range(matrix.shape[0]):
        if i!=key1 or i!=key2:
            matrix[i][key1] = min(matrix[i][key1], matrix[i][key2]) # let i be the
            matrix[key1][i] = matrix[i][key1]
        matrix[i][key2] = n
        matrix[key2][i] = n
    np.fill_diagonal(matrix, n)
    print("matrix_new: \n ", matrix)
    return matrix


def hierarchical_nail(number_of_clusters):

    # DATA
    m = number_of_clusters
    iris_data = datasets.load_iris()
    X = iris_data.data
    # X = np.array([[0.4, 0.53], [0.22, 0.38], [0.35, 0.32], [0.26, 0.19], [0.08, 0.41], [0.45 ,0.30]])
    #X = np.array([[1,6],[1,4],[1,1],[5,1]])

    # MATRIX
    matrix = compute_matrix(X)
    print(matrix)

    # # Clusters as List
    # clusters = [[i] for i in range(1, len(X)+1)]
    # # print("lenght of clusters: ", len(clusters))

    # CLUSTERS
    clusters = {}
    for i in range(len(X)):
        clusters[str(i)]=[i]
    print(clusters) # {'0': [0], '1': [1], '2': [2], '3': [3]}

    # KEYS
    key1, key2 = new_cluster(matrix) # every time should get back 2 keys
    key1 = str(int(key1))
    key2 = str(int(key2))
    print("keys: ", key1, key2)

    clusters[key1] = clusters[key1] + clusters[key2]
    del clusters[key2]


    # # updating the list cluster
    # if len(clusters[key2]) == 1:
    #     clusters[key1].append(clusters[key2][0])
    # else:
    #     clusters[key1].extend(clusters[key2])
    # del clusters[key2]
    # print(clusters)

    l = len(clusters)

    matrix_new = reformat_matrix(matrix, key1, key2)


    while l > m:
        print("in the loop")
        key1, key2 = new_cluster(matrix_new)
        key1 = str(int(key1))
        key2 = str(int(key2))
        print("keys: ", key1, key2)

        # # updating the list cluster
        # if len(clusters[key2]) == 1:
        #     clusters[key1].append(clusters[key2][0])
        # else:
        #     clusters[key1].extend(clusters[key2])
        # del clusters[key2]

        clusters[key1] = clusters[key1] + clusters[key2]
        del clusters[key2]

        # print clusters
        print("clusters: ", clusters)

        # update the matrix
        matrix_new = reformat_matrix(matrix_new, key1, key2)

        # remove one cluster from the list
        l = l - 1


    return clusters


# call the hierarchical
#
print(hierarchical_nail(2))

