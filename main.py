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
import Hierarchical
import DBSCAN
import k_means

# check the algorithms
from sklearn import datasets


def main():
    # Load the data
    # df = Load_the_data.load_the_data()

    # # ---- to be changed - start ----
    # selected_columns = df[["x", "y", "z"]]
    # df = selected_columns.copy()
    # df = df.astype(float)
    # dt = selected_columns.copy().astype(float).to_numpy() # it is an array
    # # print(df)
    # # print(dt)
    # # ---- to be changed - end ----

    # Alternative datasets from sklearn
    iris_data = datasets.load_iris()
    X = iris_data.data
    # print(dt.shape) # (150,4)

    # call k-means
    # k_means.k_means(df)

    # call DBSCAN
    # TODO: set the two parameters for dbscan
    clustering = DBSCAN.dbscan(X, 5, 1)
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)
    plt.scatter(X_transformed[:,0], X_transformed[:,1], c=clustering)
    plt.show()




if __name__ == "__main__":
    main()

