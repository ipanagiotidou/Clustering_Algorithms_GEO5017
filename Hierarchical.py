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
# TODO: BE VERY CAREFUL: THE KDTREE RETURNS ITSELFT AS WELL.
# TODO: απαγορεύεται να βάλεις απόσταση στο key του dictionary! Ξέχνα το.

# create my data
my_data = np.array([[9,49], [24,54], [51,28], [81,54], [81,23], [86,32]])
# print(my_data)

def compute_table_A(my_data):
    # create the matrix
    i = len(my_data)
    matrix = np.zeros(shape=(i,i))

    # fill the matrix
    kdtree = scipy.spatial.KDTree(my_data)
    for id1, cl1 in enumerate(my_data):
        print("\n")
        print("id1: ", id1)
        d, indices = kdtree.query(cl1, k=len(my_data), p=2)
        print("distances: ", d, "indices: ", indices) # dist:  <class 'numpy.ndarray'> index:  <class 'numpy.ndarray'>
        index_d = 0
        for index in indices:
            print("index: ", index)
            print("id: ", id1, "index: ", index, "distance: ", d[index])
            matrix[id1][index] = round(d[index_d])
            index_d+=1
    print(matrix)
    return True

compute_table_A(my_data)








# If we consider every row of this dataframe as a cluster, we Compute the distances of all the possible cluster combinations.



















