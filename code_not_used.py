

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