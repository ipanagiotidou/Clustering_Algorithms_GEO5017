

# --- --- --- --- --- --- TODO: HIERARCHICAL CODE --- --- --- --- --- ---

# TODO: delete later. Play datasets.
# X = np.array([[0.4, 0.53], [0.22, 0.38], [0.35, 0.32], [0.26, 0.19], [0.08, 0.41], [0.45 ,0.30]])
# X = np.array([[1,6],[1,4],[1,1],[5,1]])
# TODO: delete the above if not used.

# # TODO: delete later if not used: Clusters as List
# clusters = [[i] for i in range(1, len(X)+1)]
# # print("lenght of clusters: ", len(clusters))
# TODO: delete the above if not used.

# TODO: delete if not used. Was used when I had the clusters as LIST. Now I use DICTIONARIES.
# # updating the list cluster
# if len(clusters[key2]) == 1:
#     clusters[key1].append(clusters[key2][0])
# else:
#     clusters[key1].extend(clusters[key2])
# del clusters[key2]
# print(clusters)
# TODO: delete the above if not used.

# TODO: delete the above if not used.
# list_clusters = []
# list_cluster_1 = []
# list_cluster_1.append(clusters['0'])
# list_cluster_2 = []
# list_cluster_1.append(clusters['100'])
# print("list-1: ", list_cluster_1)
# print("list_clusters: ", list_cluster_1)
# list_clusters.append(list_cluster_1)
# list_clusters.append(list_cluster_2)
# TODO: delete the above if not used.

# TODO: PLOT THE RESULTS
# TODO: --- DOESN'T WORK --- Plotting the results
# # filter rows of original data
# filtered_label2 = df['cluster' == 0.0]
# print("df: \n", filtered_label2)
# # plt.scatter(filtered_label2[:, 0], filtered_label2[:, 1], color='red')
# plt.show()
# # TODO: plot
# fig = plt.figure()
# # declare axes: axes3D
# ax = fig.add_subplot(projection='3d')
#
# # show clustering result
# color_mark = ['or', 'oc', 'og', 'ob', 'oy', 'om']  # Markers of different clusters 'or' --> 'o'circle，'r'red，'b':blue
# count = 0
# for cluster_id in clusters.keys():
#     objects = clusters[cluster_id] # for every key, save the values in the objects variable
#     for i in objects: # then traverse the values of every key
#         print(i)
#         pt=[0,0,0]
#         pt[0]=float(X[i:i + 1,0])
#         pt[1] =float(X[i:i + 1,1])
#         pt[2] =float(X[i:i + 1,2])
#         print(pt)
# #         ax.plot(pt[0], pt[1], pt[2],color_mark[count])  # draw points in each cluster
# #     count=count+1
# # plt.show()
# # show_results(5,X,clusters)
# TODO: PLOT THE RESULTS

# TODO: Taken from K-means
# # TODO TASK: save in the initial dataframe the label of the clusters that each object belongs to
# # for every key in the dictionary holding the clusters
# for cluster in clusters_objects:
#     for object in cluster:
#         print("object: ", object)
#     # # use the key and retrieve the values
#     # # since every key is a cluster --> the values will be the objects that belong to this cluster
#     # # locate the row where the object is based on its index (val) and assign to the column 'cluster' its dictionary key
#     # for val in clusters[key]:
#     #     df.loc[val, 'cluster'] = int(str(key))

# TODO: DBSCAN: delete if not used --> from previously
# df = Load_the_data.load_the_data()
# df = df[["x", "y", "z"]]
# dt = df.copy().astype(float).to_numpy() # turn the dataframe into an array
# data = load_iris()
# df = pd.DataFrame(data.data, columns=data.feature_names)
# clustering = DBSCAN.dbscan(df, 5, 1) # dataframe
# TODO: DBSCAN: delete THE ABOVE if not used --> from previously


# # # TODO: helpful for depiction
# # pca = PCA(n_components=2)
# # X_transformed = pca.fit_transform(X)
# # plt.scatter(X_transformed[:,0], X_transformed[:,1], c=clustering)
# # plt.show()

# TODO: play dataframe for evaluation
# Create a PLAY DataFrame
# data = {'label': ['building', 'car', 'building', 'fence', 'pole', 'building', 'tree' ], 'cluster': [0,1,4,2,3,0, 2]}
# play_df = pd.DataFrame(data)
# # print(play_df)


# TODO: EVALUATION - works only for 5 clusters
# accuracies.append(matrix[0][column_indices[0]] / 100)
# accuracies.append(matrix[1][column_indices[1]] / 100)
# accuracies.append(matrix[2][column_indices[2]] / 100)
# accuracies.append(matrix[3][column_indices[3]] / 100)
# accuracies.append(matrix[4][column_indices[4]] / 100)

# # Create a PLAY DataFrame
# data = {
#     'label': ['building', 'car', 'building', 'fence', 'pole', 'building', 'tree', 'fence', 'fence', 'fence', 'fence',
#               'pole',
#               'building', 'building', 'car', 'car', 'car', 'car', 'pole', 'pole', 'pole', 'tree', 'tree', 'tree',
#               'tree'],
#     'cluster': [0, 1, 4, 2, 3, 0, 2, 4, 4, 4, 4, 4, 0, 0, 1, 2, 4, 4, 4, 3, 2, 3, 3, 4, 3]}
# play_df = pd.DataFrame(data)
# # print(play_df)
# error_matrix = EVALUATION.create_error_matrix(play_df, number_of_clusters=5)

# TODO: code from evaluation
# Now, choose the maximum value per row and divide with 100 --> number of objects for buildings, cars, fences, trees, poles.
# Find the column index which has the maximum value for each row
# column_indices = matrix_df.idxmax(axis=1)
indices = []
matrix_copy = np.copy(matrix)
for i in range(number_of_clusters):  # do as many times as the number of clusters
    result = np.where(matrix_copy == np.amax(matrix_copy))
    listOfCordinates = list(zip(result[0], result[1]))
    indices.append(listOfCordinates[0])
    matrix_copy[
        listOfCordinates[0][0]] = 0  # after that this row and column will never be visited again (category and cluster)
    matrix_copy[:, listOfCordinates[0][1]] = 0
    # print(matrix)
# print("indices: ", indices)
accuracies = []
for j, k in indices:  # number of clusters
    # print("j,k:", j,k)
    # print(matrix[j][k])
    accuracies.append(matrix[j][k] / 100)
    if j == 0:
        print("building:", matrix[j][k] / 100)
    if j == 1:
        print("car:", matrix[j][k] / 100)
    if j == 2:
        print("fence:", matrix[j][k] / 100)
    if j == 3:
        print("tree:", matrix[j][k] / 100)
    if j == 4:
        print("pole:", matrix[j][k] / 100)

# accuracy_matrix = {'label': ['building', 'car', 'fence', 'tree', 'pole'], 'accuracy': [i for i in accuracies]}
# accuracy_matrix = pd.DataFrame(accuracy_matrix)
# print(accuracy_matrix)
# TODO: code from evaluation - end


# TODO: ALL COMBINATIONS
# df_feat = df[["density_3d", "volume", "height"]] # OK
# df_feat = df[["density_3d", "volume", "density_2d"]] # NOT
# df_feat = df[["density_3d", "volume", "area_3d"]] # NOT
# df_feat = df[["density_3d", "volume", "proj_area"]] # NOT
# df_feat = df[["density_3d", "height", "density_2d"]] # COULD BE
# df_feat = df[["density_3d", "height", "area_3d"]] # COULD BE
# df_feat = df[["density_3d", "height", "proj_area"]] #  COULD BE
# df_feat = df[["density_3d", "area_3d", "proj_area"]] # NOT
# TODO: ALL COMBINATIONS finished