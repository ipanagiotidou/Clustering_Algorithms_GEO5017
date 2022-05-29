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
import calculate_features


# check the algorithms
from sklearn import datasets


def main():
    # TODO SOS: POSSIBLY I HAVE TO CUT THE DATASET IN THE ALGORITHM
    # TODO SOS: SO I SHOULD PROVIDE THE WHOLE DATASET IN THE FUNCTION, AND THE FUNCTION CUTS THE DATASET/

    # TODO: Load the data - CHECK ABOVE 'TODO SOS'
    # # ---- to be changed - start ----
    # df = Load_the_data.load_the_data()
    # # selected_columns = df[["x", "y", "z"]]
    # df = df[["x", "y", "z"]]
    # dt = df.copy().astype(float).to_numpy() # turn the dataframe into an array
    # # # ---- to be changed - end ----


    # # TODO: calculate features
    # df = calculate_features.calculate_features()
    # df = df[["volume", "proj_area", "area_3d", "height", "density_2d","density_3d"]]


    # # TODO: --- --- --- --- call K-MEANS algorithm --- --- --- ---
    # df = calculate_features.calculate_features()
    # df_nl = df[['bname', 'label']]
    # k_means.k_means(df_nl, df_feat=df[["volume", "proj_area", "area_3d", "height", "density_2d","density_3d"]]) # I need the dataframe in k-means


    # # # TODO: --- --- --- --- call DBSCAN --- --- --- ---
    # # # TODO TASK: SET the two parameters for dbscan
    # df = calculate_features.calculate_features()
    # df_feat = df[["volume", "proj_area", "area_3d", "height", "density_2d", "density_3d"]]
    # df_nl = df[['bname', 'label']]
    # minPts = 5
    # epsilon = 15
    # DBSCAN.dbscan(df_nl, df_feat, minPts, epsilon)


    # # # TODO: --- --- --- --- call HIERARCHICAL clustering --- --- --- ---
    # # # TODO: X can be a dataframe since we turn the dataframe into an array when we compute the MATRIX
    # TOdO: pass the dataframe only with features: df = Load_the_data.load_the_data() --> df = df[["x", "y", "z"]]
    # df = calculate_features.calculate_features()
    # df_feat = df[["volume", "proj_area", "area_3d", "height", "density_2d", "density_3d"]]
    # df_nl = df[['bname', 'label']]
    # # TODO SOS : take care of the n = 10**2 value !!!!
    # clustering = hierarchical_Nail.hierarchical_nail(df_nl, df_feat, 5) # pass the dataframe, only with features

    pass







if __name__ == "__main__":
    main()


