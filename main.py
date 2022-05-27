import numpy as np
import pandas as pd
import random
import math
import scipy.spatial
from sklearn.datasets import load_iris
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
import hierarchical_Nail


# check the algorithms
from sklearn import datasets


def main():
    # TODO SOS: POSSIBLY I HAVE TO CUT THE DATASET IN THE ALGORITHM
    # TODO SOS: SO I SHOULD PROVIDE THE WHOLE DATASET IN THE FUNCTION, AND THE FUNCTION CUTS THE DATASET/

    # TODO: Load the data - CHECK ABOVE 'TODO SOS'
    # # ---- to be changed - start ----
    df = Load_the_data.load_the_data()
    # # selected_columns = df[["x", "y", "z"]]
    # df = df[["x", "y", "z"]]
    # dt = df.copy().astype(float).to_numpy() # turn the dataframe into an array
    # # # ---- to be changed - end ----

    # # TODO: --- --- --- --- call K-MEANS algorithm --- --- --- ---
    df = Load_the_data.load_the_data()
    # k_means.k_means(df) # I need the dataframe in k-means

    # # # TODO: --- --- --- --- call DBSCAN --- --- --- ---
    # # # TODO TASK: SET the two parameters for dbscan
    # df = Load_the_data.load_the_data()
    # df = df[["x", "y", "z"]]
    # dt = df.copy().astype(float).to_numpy() # turn the dataframe into an array
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    clustering = DBSCAN.dbscan(df, 5, 1) # dataframe


    # # # TODO: helpful for depiction
    # # pca = PCA(n_components=2)
    # # X_transformed = pca.fit_transform(X)
    # # plt.scatter(X_transformed[:,0], X_transformed[:,1], c=clustering)
    # # plt.show()

    # # # TODO: --- --- --- --- call HIERARCHICAL clustering --- --- --- ---
    # # # TODO: X can be a dataframe since we turn the dataframe into an array when we compute the MATRIX
    # TOSO: pass the dataframe only with features: df = Load_the_data.load_the_data() --> df = df[["x", "y", "z"]]
    # df = Load_the_data.load_the_data()
    # df = df[["x", "y", "z"]]
    # clustering = hierarchical_Nail.hierarchical_nail(df, 5) # pass the dataframe, only with features
    # # # TODO TASK: return the dataframe 'df' with the labels !






if __name__ == "__main__":
    main()

