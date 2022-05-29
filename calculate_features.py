# Modules
import Load_the_data
# # 1 # CALCULATE THE CONVEX HULL and VOLUME
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, distance_matrix
import os
import pandas as pd
import statistics


def calculate_features():
    global label
    files = os.listdir("C:/Users/Panagiotior/Desktop/Q3/MachineLearning/hw/GEO5017-A1-Clustering/scene_objects/scene_objects/data")
    path = "C:/Users/Panagiotior/Desktop/Q3/MachineLearning/hw/GEO5017-A1-Clustering/scene_objects/scene_objects/data/"

    # create the BIG dataframe to store the features of all the objects
    df = pd.DataFrame(columns=['bname', 'label'])  # set the column names

    for file in files:
        file_extension = os.path.splitext(file)[-1].lower() # search for files ending in .xyz
        base_name = os.path.splitext(file)[-2].lower() # reads the 3-digits integer base name, e.g. 499
        iname = int(base_name) # convert the base_name into an integer # TODO: make sure it works for the first base names, e.g. 001, 002, ...
        if file_extension == ".xyz":
            # determine the label name based on the base_name of the file, e.g. from 0 (integer of 000) to 099 it is a building.
            if (iname >= 0 and iname < 100):
                label = 'building'
            elif (iname >= 99 and iname < 200):
                label = 'car'
            elif (iname >= 200 and iname < 300):
                label = 'fence'
            elif (iname >= 300 and iname < 400):
                label = 'pole'
            elif (iname >= 400 and iname < 500):
                label = 'tree'
            else:
                label = 'FALSE'

            # Creating the basic dataframe
            df_row = pd.DataFrame({"bname": [iname], "label": [label]})

            list_z = []
            pts_list = []
            pts_list_xy = []
            with open(path + file ) as f:
                lines = f.readlines()
                for line in lines:
                    attributes = line.split()
                    pts_list.append([float(attributes[0]),float(attributes[1]),float(attributes[2])])
                    pts_list_xy.append([float(attributes[0]), float(attributes[1])])
                    list_z.append(float(attributes[2]))
                        # the .xyz file in a numpy array
                pts = np.array(pts_list)
                pts_xy = np.array(pts_list_xy)
            # print(pts)

            # -- -- -- -- -- -- 1st FEATURE: VOLUME -- -- -- -- -- -- --
            # create the convex hull
            hull = ConvexHull(pts)
            # Calculate volume
            volume = hull.volume
            # Add volume in the object's dataframe
            df_row.loc[:, 'volume'] = volume

            # # PLOT THE CONVEX HULL
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection="3d")
            # # Plot defining corner points
            # ax.plot(pts.T[0], pts.T[1], pts.T[2], "ko")
            # # 12 = 2 * 6 faces are the simplices (2 simplices per square face)
            # for s in hull.simplices:
            #     s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            #     ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")
            # # Make axis label
            # for i in ["x", "y", "z"]:
            #     eval("ax.set_{:s}label('{:s}')".format(i, i))
            # plt.show()
            # # END OF PLOT THE CONVEX HULL



            # -- -- -- -- -- --2nd FEATURE: PROJECTION AREA -- -- -- -- -- -- --
            hull_xy = ConvexHull(pts_xy)
            # Area: Surface area of the convex hull when input dimension > 2. When input points are 2-dimensional, this is the perimeter of the convex hull.
            proj_area = hull_xy.area
            # Add volume in the object's dataframe
            df_row.loc[:, 'proj_area'] = proj_area

            # -- -- -- -- -- --3rd FEATURE: PROJECTION AREA -- -- -- -- -- -- --
            hull = ConvexHull(pts)
            area_3d = hull.area
            # Add volume in the object's dataframe
            df_row.loc[:, 'area_3d'] = area_3d

            # TODO: IF YOU HAVE TIME, CORRECT THE HEIGHT. THE MAX OR MIN values, if based in a single value, might be OUTLIERS.
            # -- -- -- -- -- --4th FEATURE: HEIGHT -- -- -- -- -- -- --
            median_z = statistics.median(list_z)
            height = median_z
            # Add volume in the object's dataframe
            df_row.loc[:, 'height'] = height

            # TODO: The point density is in 2D. What are the benefits if I calculate it in 3D ?
            # -- -- -- -- -- --5th FEATURE: POINT DENSITY 2D -- -- -- -- -- -- --
            density_2d = proj_area / len(pts_list)
            # Add volume in the object's dataframe
            df_row.loc[:, 'density_2d'] = density_2d

            # -- -- -- -- -- --6th FEATURE: POINT DENSITY 3D -- -- -- -- -- -- --
            density_3d = area_3d / len(pts_list)
            # Add volume in the object's dataframe
            df_row.loc[:, 'density_3d'] = density_3d

            # AFTER CALCULATING ALL THE FEATURES --> concatenate the row (object) to the BIG dataframe
            df = pd.concat([df, df_row], sort=False, ignore_index=True)


    # print(df)

    return df

print(calculate_features())