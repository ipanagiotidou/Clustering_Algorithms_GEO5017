import numpy as np


# The aim is to create a matrix which will hold as rows the 5 categories of objects, and as columns the cluster's_ids
# In the cells, I will fill with the number of occurence of every category in each cluster_id
# For example, 50 buildings in cluster 0.
import pandas as pd


def create_error_matrix(df, number_of_clusters):
    n = number_of_clusters

    matrix = np.zeros((5,n)) # 5 rows for the 5 objects, number of columns --> number of clusters

    # iterate through the objects (rows) and clusters (columns)
    for j in range(n): # traverse the columns and fill the whole column
        val1 = (df[(df.label == 'building') & (df.cluster == j)]).count(axis=1)
        matrix[0][j] = len(val1)
        val2 = (df[(df.label == 'car') & (df.cluster == j)]).count(axis=1)
        matrix[1][j] = len(val2)
        val3  = (df[(df.label == 'fence') & (df.cluster == j)]).count(axis=1)
        matrix[2][j] = len(val3)
        val4 = (df[(df.label == 'tree') & (df.cluster == j)]).count(axis=1)
        matrix[3][j] = len(val4)
        val5 = (df[(df.label == 'pole') & (df.cluster == j)]).count(axis=1)
        matrix[4][j] = len(val5)


    matrix_df = pd.DataFrame(matrix, columns=[i for i in range(n)], index=['building', 'car', 'fence', 'tree', 'pole'])
    print("matrix_df: \n", matrix_df)



    return 0

