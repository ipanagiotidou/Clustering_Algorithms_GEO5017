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
                # Manhatan distance --> Replace with Euclidean
                matrix[i][j] = sum(abs(val1-val2) for val1, val2 in zip(my_data[i],my_data[j]))
                matrix[j][i] = sum(abs(val1 - val2) for val1, val2 in zip(my_data[i], my_data[j]))
            else:
                matrix[i][j] = 10**2 # change

    #matrix_b = np.triu(matrix)
    #print("matrix_b: ", matrix_b)
    print(matrix)
    return matrix # matrix_a

# this function computes the distances and returns the clusters in a list. [[], []]
def new_cluster(matrix):
    # function that takes a matrix and chooses which two clusters will form a new cluster
    clusters = []
    indices = np.where(matrix == np.amin(matrix))
    print(indices)
    clusters.append(str(indices[0][0]))
    clusters.append(str(indices[1][0]))
    # remove the two clusters from the previous cluster
    return clusters[0], clusters[1]

def hierarchical(number_of_clusters):
    m = number_of_clusters
    # create data
    X = np.array([[1,6,5], [1,4,5], [1,1,5], [5,1,5]])
    # X = np.array([[0.4, 0.53], [0.22, 0.38], [0.35, 0.32], [0.26, 0.19], [0.08, 0.41], [0.45 ,0.30]])
    # calculate the distance matrix
    matrix = compute_matrix(X)
    # print(matrix)

    clusters = {}
    for i in range(len(X)):
        clusters[str(i)]=[i]
    print(clusters) # {'0': [0], '1': [1], '2': [2], '3': [3]}

    key1, key2 = new_cluster(matrix) # every time should get back 2 keys
    print("key1: ", key1, type(key1))
    print("key2: ", key2, type(key2))
    clusters[key1] = clusters[key1] + clusters[key2]
    del clusters[key2]
    # print(clusters) # {'0': [0, 1], '2': [2], '3': [3]}
    l = len(clusters)

    while l > m:
        print("mpika")
        list_all_el_val_dists = []
        # print("clusters.values(): " , clusters.values())
        for cl1 in clusters.keys(): # dict_values([[0, 1], [2], [3]])
            print("cl1: ", cl1)
            cl1_indices = clusters[cl1] # [0, 1] type: list
            i0 = cl1_indices[0]
            objects_1=[]
            objects_1.append(float(X[i0:i0 + 1, 0]))
            objects_1.append(float(X[i0:i0 + 1, 1]))
            objects_1.append(float(X[i0:i0 + 1, 2]))

            print("objects_1: ", objects_1)

        #
        #     for cl2 in clusters.keys():
        #
        #         if cl1 != cl2:
        #             print("val2: ", cl2)
        #             # # print(type(val2))
        #             # print("clusters[val1]: ", clusters[val1])
        #             # for cl1 in clusters[val1]:
        #
        #
        #
        #             # Lan's function
        #             dist= scipy.spatial.distance.cdist(clusters[val1], clusters[val2], 'euclidean')
        #             cur_min_dist=np.min(dist)
        #             list_all_el_val_dists.append([val1, val2, cur_min_dist])
        #
        #
        #
        # dt = pd.DataFrame(list_all_el_val_dists)
        # # # print("list_all_el_val_dists: ", list_all_el_val_dists)
        # print(dt)

        # get the row index position of min value in column 2(where I keep the distances)
        # minValueRowIndexObj = dt[2].idxmin()
        # print("min values of columns are at row index position :")
        # print(minValueRowIndexObj)
        # # Now, I should find in which key the value is and retrieve the key to use it to replace
        # key = [k for k, v in clusters.items() if minValueRowIndexObj in v] # clusters.items(): dict_items([('0', [0, 1]), ('2', [2]), ('3', [3])])
        # print("keys: ", key) # returns the key in a list, e.g. ['0']
        # # Now I should take this key and replace it with the key-values it had and also the
        # key_1 = key[0]
        # print("key_1: ", key_1, type(key_1))
        # key_2 = str(dt[minValueRowIndexObj][1]) # the value I want to import, which is the row index
        # print("key_2: ", key_2, type(key_2))
        # # do the magic
        # clusters[key_1] = clusters[key_1] + clusters[key_2]
        # del clusters[key_2]


        l = l - 1
    return clusters


# call the hierarchical
print(hierarchical(2))

