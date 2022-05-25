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

### -------------------------------------------- HIERARCHICAL clustering algorithm -------------------------------------
# TODO: Hierarchical algorithm
# TODO: Change 10**2 value

def compute_matrix(my_data):
    # create the REFERENCE MATRIX
    l = len(my_data)
    matrix = np.zeros(shape=(l,l))
    matrix_b = np.zeros(matrix.shape)
    # fill the matrix
    for i in range(l):
        for j in range(l):
            if i!=j:
                # if i > j:
                #     matrix[i][j] = 10**2 # change
                # else:
                    # Manhatan distance --> Replace with Euclidean
                matrix[i][j] = sum(abs(val1-val2) for val1, val2 in zip(my_data[i],my_data[j]))
            else:
                matrix[i][j] = 10**2 # change

    #matrix_b = np.triu(matrix)
    #print("matrix_b: ", matrix_b)
    #print(matrix)
    return matrix # matrix_a

def reformat_matrix(matrix, key1, key2):
    # TODO - question: can it be that key2 > key1 --> the way I return the keys in the function 'new_cluster' ?

    key1 = int(key1)
    key2 = int(key2)
    size = matrix.shape[0]-1
    matrix_new = np.delete(matrix, key2, 1) # key2 for the row --> deletes row
    matrix_new = np.delete(matrix_new, key2, 0)
    #print("matrix_new: ", matrix_new)
    for i in range(matrix.shape[0]-1):
        if i!=key1:
            print("i,", i)
            matrix_new[i][key1] = min(matrix[i][key1], matrix[i][key2]) # let i be the
    print("matrix_new:\n ", matrix_new)
    return matrix_new


def distance_samples(s1, s2, matrix):
    dist = matrix[s1][s2]
    if dist == 10 ** 2: # change
        # change the order of the el1, el2 to find the distance and assign the new values to el1 = el2
        dist = matrix[s2][s1]
        # temp = s1
        # s1 = s2
        # s2 = temp
        row = [s2, s1, dist]
        return row
    else:
        row = [s1, s2, dist]
    return row

# this function computes the distances and returns the clusters in a list. [[], []]
def new_cluster(matrix):
    # function that takes a matrix and chooses which two clusters will form a new cluster
    clusters = []
    # print(matrix)
    indices = np.where(matrix == np.amin(matrix))
    print("indices:",indices)
    clusters.append(str(indices[0][0]))
    clusters.append(str(indices[1][0]))
    # remove the two clusters from the previous cluster
    # if (clusters[0] < clusters[1]):
    #     return clusters[0], clusters[1]
    # else:
    #     return clusters[1], clusters[0]
    return clusters[0], clusters[1]


def hierarchical(number_of_clusters):
    m = number_of_clusters
    # create data
    # X = np.array([[1,6], [1,4], [1,1], [5,1]])
    X = np.array([[0.4, 0.53], [0.22, 0.38], [0.35, 0.32], [0.26, 0.19], [0.08, 0.41], [0.45 ,0.30]])
    # calculate the distance matrix
    matrix = compute_matrix(X)
    # print(matrix)

    clusters = {}
    for i in range(len(X)):
        clusters[str(i)]=[i]
    print(clusters) # {'0': [0], '1': [1], '2': [2], '3': [3], '4': [4], '5': [5]}

    key1, key2 = new_cluster(matrix) # every time should get back 2 keys
    print("key1: ", key1, type(key1))
    print("key2: ", key2, type(key2))

    clusters[key1] = clusters[key1] + clusters[key2] # {'0': [0], '1': [1], '2': [2,5], '3': [3], '4': [4]}
    del clusters[key2]
    print(clusters)
    l = len(clusters)

    matrix_new = reformat_matrix(matrix, key1, key2)


    while l > m:
        key1, key2 = new_cluster(matrix_new)
        print("keys: ", key1, key2)
        clusters[key1] = clusters[key1] + clusters[key2]
        del clusters[key2]
        print("clusters: ", clusters)
        matrix_new = reformat_matrix(matrix_new, key1, key2)
        l = l - 1


    return clusters


# call the hierarchical
print(hierarchical(2))

