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
                if i > j:
                    matrix[i][j] = 10**2 # change
                else:
                    # Manhatan distance --> Replace with Euclidean
                    matrix[i][j] = sum(abs(val1-val2) for val1, val2 in zip(my_data[i],my_data[j]))
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

def distance_samples(s1, s2, matrix):
    dist = matrix[s1][s2]
    if dist == 10 ** 2: # change
        return # None
    else:
        row = [s1, s2, dist]
    return row

def distance_clusters(cluster1, cluster2, matrix):
    el_val_dist_list = []
    for el1 in cluster1:
        for el2 in cluster1:
            dist = matrix[el1][el2]
            if dist == 10 ** 2:  # change
                return # None
            else:
                el_val_dist_list.append([el1, el2, dist])
    return el_val_dist_list

def distance_sample_cluster(cluster, value, matrix):
    # it should return [el, val, dist]
    el_val_dist_list = []
    print(list)
    for el in cluster:  # first time will be el = 0 and then el = 1
        dist = matrix[el][value] # row = el, col = val # first time matrix[0][2] -> 5, matrix[1][2] -> 3
        if dist == 10**2: # change
            return None
        else:
            el_val_dist_list.append([el, value, dist]) # [0, 2, 5], [1, 2, 3]
    return el_val_dist_list



def hierarchical(number_of_clusters):
    m = number_of_clusters
    # create data
    X = np.array([[1,6], [1,4], [1,1], [5,1]])
    # calculate the distance matrix
    matrix = compute_matrix(X)
    # print(matrix)

    clusters = {}
    for i in range(len(X)):
        clusters[str(i)]=[i]
    # print(clusters) # {'0': [0], '1': [1], '2': [2], '3': [3]}

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
        # empty your frames
        row = []
        el_clusters_list = []
        el_val_dist_list_1 = []
        el_val_dist_list_2 = []
        for val1 in clusters.values(): # dict_values([[0, 1], [2], [3]])
            # print("val1: ", val1)
            for val2 in clusters.values():

                if val1 != val2:
                    # print("val2: ", val2)
                    # print(type(val2))

                    # 4 CASES: two of them are the same

                    # Case 1: both clusters/lists
                    if len(val1) > 1 and len(val2) > 1:
                        # call
                        print("case1")
                        el_clusters_list = distance_clusters(val1, val2, matrix)
                        if el_clusters_list == None:
                            pass  # not sure it should be a pass or continue
                        else:
                            list_all_el_val_dists.extend(el_clusters_list)


                    # Case 2: one sample (val1) and one cluster/list (val2)
                    if not len(val1) > 1:
                        # print("length val1: ", val1)
                        if len(val2) > 1:
                            print("case2")
                            # print("val2: ", val2)
                            # call the distance_sample_cluster
                            el_val_dist_list_1 = distance_sample_cluster(val1, val2[0], matrix)
                            if el_val_dist_list_1 == None:
                                pass
                            else:
                                list_all_el_val_dists.extend(el_val_dist_list_1)
                            print(el_val_dist_list_1)


                    # Case 3: # one cluster/list (val1) and one value/sample (val2)
                    if not len(val2) > 1:
                        if len(val1) > 1:
                            print("case3")
                            # call the distance_sample_cluster
                            el_val_dist_list_2 = distance_sample_cluster(val1, val2[0], matrix)
                            if el_val_dist_list_2 == None:
                                pass
                            else:
                                list_all_el_val_dists.extend(el_val_dist_list_2)
                            print(el_val_dist_list_2)

                    # Case 4: # both samples
                    if not len(val1) > 1 and not len(val2) > 1: # μήπως έχει σημασία ποιο είναι μικρότερο ή μεγαλύτερο κάποια στιγμή;
                        # call the simple distance_function
                        print("case4")
                        # print("val1:", val1)
                        # print("val2:", val2)
                        row = distance_samples(val1[0], val2[0], matrix)
                        if row == None:
                            pass
                        else:
                            list_all_el_val_dists.append(row)
                        print(row)

        dt = pd.DataFrame(list_all_el_val_dists)
        # # print("list_all_el_val_dists: ", list_all_el_val_dists)
        print(dt)

        # get the row index position of min value in column 2(where I keep the distances)
        minValueRowIndexObj = dt[2].idxmin()
        print("min values of columns are at row index position :")
        print(minValueRowIndexObj)
        # Now, I should find in which key the value is and retrieve the key to use it to replace
        key = [k for k, v in clusters.items() if minValueRowIndexObj in v] # clusters.items(): dict_items([('0', [0, 1]), ('2', [2]), ('3', [3])])
        print("keys: ", key) # returns the key in a list, e.g. ['0']
        # Now I should take this key and replace it with the key-values it had and also the
        key_1 = key[0]
        print("key_1: ", key_1, type(key_1))
        key_2 = str(dt[minValueRowIndexObj][1]) # the value I want to import, which is the row index
        print("key_2: ", key_2, type(key_2))
        # do the magic
        clusters[key_1] = clusters[key_1] + clusters[key_2]
        del clusters[key_2]


        l = l - 1
    return clusters


# call the hierarchical
print(hierarchical(2))

