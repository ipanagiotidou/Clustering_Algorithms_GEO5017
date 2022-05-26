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

### ------ Modules -------
import Load_the_data
# import Hierarchical
import DBSCAN
import k_means
import hierarchical_Nail


# check the algorithms
from sklearn import datasets


def main():
    # Load the data
    # df = Load_the_data.load_the_data()

    # # ---- to be changed - start ----
    df = Load_the_data.load_the_data()
    selected_columns = df[["x", "y", "z"]]
    df = selected_columns.copy()
    df = df.astype(float)
    dt = selected_columns.copy().astype(float).to_numpy() # it is an array
    X = dt

    # # print(df)
    # # print(dt)
    # # ---- to be changed - end ----

    # Alternative datasets from sklearn
    # iris_data = datasets.load_iris()
    # X = iris_data.data

#     X =[[1.06, 9.2, 151, 54.4, 1.6, 9077, 0, 0.628],
# [0.89, 10.3, 202, 57.9, 2.2, 5088, 25.3, 1.555],
# [1.43, 15.4, 113, 53, 3.4, 9212, 0, 1.058],
# [1.02, 11.2, 168, 56, 0.3, 6423, 34.3, 0.7],
# [1.49, 8.8, 192, 51.2, 1, 3300, 15.6, 2.044],
# [1.32, 13.5, 111, 60, -2.2, 11127, 22.5, 1.241],
# [1.22, 12.2, 175, 67.6, 2.2, 7642, 0, 1.652],
# [1.1, 9.2, 245, 57, 3.3, 13082, 0, 0.309],
# [1.34, 13, 168, 60.4, 7.2, 8406, 0, 0.862],
# [1.12, 12.4, 197, 53, 2.7, 6455, 39.2, 0.623],
# [0.75, 7.5, 173, 51.5, 6.5, 17441, 0, 0.768],
# [1.13, 10.9, 178, 62, 3.7, 6154, 0, 1.897],
# [1.15, 12.7, 199, 53.7, 6.4, 7179, 50.2, 0.527],
# [1.09, 12, 96, 49.8, 1.4, 9673, 0, 0.588],
# [0.96, 7.6, 164, 62.2, -0.1, 6468, 0.9, 1.4],
# [1.16, 9.9, 252, 56, 9.2, 15991, 0, 0.62],
# [0.76, 6.4, 136, 61.9, 9, 5714, 8.3, 1.92],
# [1.05, 12.6, 150, 56.7, 2.7, 10140, 0, 1.108],
# [1.16, 11.7, 104, 54, -2.1, 13507, 0, 0.636],
# [1.2, 11.8, 148, 59.9, 3.5, 7287, 41.1, 0.702],
# [1.04, 8.6, 204, 61, 3.5, 6650, 0, 2.116],
# [1.07, 9.3, 174, 54.3, 5.9, 10093, 26.6, 1.306]]
    # print(dt.shape) # (150,4)

    # call k-means
    # k_means.k_means(df)

    # # call DBSCAN
    # # TODO: set the two parameters for dbscan
    # clustering = DBSCAN.dbscan(X, 5, 1)
    # pca = PCA(n_components=2)
    # X_transformed = pca.fit_transform(X)
    # plt.scatter(X_transformed[:,0], X_transformed[:,1], c=clustering)
    # plt.show()

    # call hierarchical clustering
    print("lenght of data: ", len(X))
    # print("data: ", X)
    clustering = hierarchical_Nail.hierarchical_nail(X, 5)
    # pca = PCA(n_components=2)
    # X_transformed = pca.fit_transform(X)
    # plt.scatter(X_transformed[:,0], X_transformed[:,1], c=clustering)
    # plt.show()
    print(clustering)




if __name__ == "__main__":
    main()

