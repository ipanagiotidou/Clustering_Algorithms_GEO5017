import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import validation_method

### -------------------------------------------- HIERARCHICAL clustering algorithm -------------------------------------
# TODO remaining TASK: MAKE SURE THIS VALUE WORKS FOR YOUR DATASET  --> --> --> --> Change 10**2 value
n = 10**2

def compute_matrix(my_data):
    my_data = np.array(my_data)
    l = len(my_data)

    # initialize an empty 2D matrix to hold the distances between the clusters
    matrix = np.zeros(shape=(l,l))

    # fill the matrix
    for i in range(l):
        for j in range(l):

            if i!=j: # calculate distance only between different clusters (skip the diagonal)

                # We calculate the Euclidean distance between all pairs of clusters
                val1 = my_data[i]
                val2 = my_data[j]
                matrix[i][j] = round(np.linalg.norm(val1-val2), 2)

            else:
                matrix[i][j] = n   # assign a big value to the diagonal elements so that we never get them back as the minimum value

    return matrix


def new_cluster(matrix):
    '''
    The function takes a matrix and returns the row and column (indices) where it found the minimum value.
    These indices indicate the objects that should be grouped in the same cluster.
    Returns the indices in ascending order, in string format since I will use them as keys to retrieve values from a dictionary.
    '''
    c = []

    indices = np.where(matrix == np.amin(matrix))
    c.append(str(indices[0][0]))
    c.append(str(indices[1][0]))

    if (c[0] < c[1]):
        return str(int(c[0])), str(int(c[1]))
    else:
        return str(int(c[1])), str(int(c[0]))


def reformat_matrix(matrix, key1, key2):
    '''
    The function takes as parameters the matrix and two indices.
    These two indices correspond to the objects that should be grouped into the same cluster.

    We reformat the Matrix of distances keeping the smallest distance between the objects and the new cluster.
    So, for example, if we have to group objects 2 and 5, we will reassign distance values to all the rest of the objects as:
    d(0,2) = min ( d(0,2), d(0,5) ).
    Finally, in the row and column of the matrix belonging to object 5 we will assign a very large value
     so that it will never be given back as the smallest distance. Meaning...we will never be forced to group it to a cluster!
    '''

    # turn the keys of the dictionary from string to integers to use them for indexing.
    key1 = int(key1)
    key2 = int(key2)
    for i in range(matrix.shape[0]): # loop by using the row dimension of the matrix
        if i!=key1 or i!=key2: # don't do the below in the rows and cols of the objects that are grouped together
            matrix[i][key1] = min(matrix[i][key1], matrix[i][key2])
            matrix[key1][i] = matrix[i][key1] # do the opposite because the matrix is symmetric

        # assign a very big number to the rows and cols of the second object (key2)
        matrix[i][key2] = n
        matrix[key2][i] = n
    # fill the diagonal again with big values
    np.fill_diagonal(matrix, n)

    # print("matrix_new: \n ", matrix)
    return matrix


def hierarchical(df_nl, df_feat, number_of_clusters):
    m = number_of_clusters

    # TODO: normalize the data = each attribute value / max possible value of this attribute
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df_feat)
    df = pd.DataFrame(x_scaled)
    # print("df_normalized: \n", df)

    # DISTANCE MATRIX:
    # holds the shortest distances between all clusters (each cluster containing an object)
    matrix = compute_matrix(df)

    # CLUSTERS:
    # Initialize a dictionary with one key-value pair for every object/cluster --> each key representing one cluster
    clusters = {}   # e.g. {'0': [0], '1': [1], '2': [2], '3': [3]} for 4 objects
    for i in range(len(df)): clusters[str(i)]=[i]
    # print(clusters)

    # KEYS:
    # this function returns the indices (row and column) of the minimum value (distance) in the matrix
    key1, key2 = new_cluster(matrix)
    # print("keys: ", key1, key2)

    # groups the clusters into one: updates the value of key1 with the values of key1 and key2
    clusters[key1] = clusters[key1] + clusters[key2]
    # delete completely the cluster of key2
    del clusters[key2]

    # Reformat the matrix with new distances between the clusters after grouping clusters
    matrix_new = reformat_matrix(matrix, key1, key2)


    l = len(clusters)
    while l > m: # exit the loop when I am left with the desired number of clusters (m) in the dictionary
        # repeat the same process where I find and connect the two clusters based on the minimum distance measure
        key1, key2 = new_cluster(matrix_new)
        clusters[key1] = clusters[key1] + clusters[key2]
        del clusters[key2]

        # update the matrix with the new distances
        matrix_new = reformat_matrix(matrix_new, key1, key2)

        # remove 1 from the iterator after grouping two clusters
        l = l - 1


    # save in the initial dataframe the label of the clusters that each object belongs to
    for key in clusters.keys(): # for every key in the dictionary --> each key representing one cluster
        for val in clusters[key]:
            # locate the row of the dictionary based on the index val and assign the column 'cluster' with the key of the dictionary
            df_nl.loc[val, 'cluster'] = int(str(key))

    # print("clusters: \n", clusters)
    # print("df: \n", df)

    cl_of_pts = list_of_names = df_nl['cluster'].to_list()

    # TODO: call EVALUATION method
    n = len(set(cl_of_pts)) # take all unique values from the list 'cl_of_pts' to find the number of clusters
    error_matrix = validation_method.create_error_matrix(df_nl, n)

    # TODO: PLOT
    X = np.array(df)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=cl_of_pts, cmap='Set1')
    plt.show()

    # return df
    return clusters






