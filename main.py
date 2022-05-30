### ------ Modules -------
import DBSCAN
import k_means
import hierarchical
import calculate_features

def main():

    # TODO: --- --- --- --- call K-MEANS algorithm --- --- --- ---
    # # todo: calculate features
    # df = calculate_features.load_and_calculate_features()
    # df_nl = df[['bname', 'label']]
    # # todo: choose for the desired features
    # df_feat = df[["density_3d", "height", "area_3d"]] # chosen
    # # df_feat = df[["density_3d", "height", "proj_area"]] #  alternative
    # # todo: call k_means function
    # df_clustered, number_of_clusters = k_means.k_means(df_nl, df_feat)


    # TODO: --- --- --- --- call DBSCAN algorithm --- --- --- ---
    # # todo: calculate features
    # df = calculate_features.load_and_calculate_features()
    # df_nl = df[['bname', 'label']]
    # # todo: choose for the desired features
    # df_feat = df[["density_3d", "height", "area_3d"]]
    # # todo task: Set the parameters for dbscan
    # minPts = 10
    # epsilon = 5
    # # todo: call the dbscan function
    # DBSCAN.dbscan(df_nl, df_feat, minPts, epsilon)


    # TODO: --- --- --- --- call HIERARCHICAL clustering --- --- --- ---
    # df = calculate_features.load_and_calculate_features()
    # df_nl = df[['bname', 'label']]
    # # todo: choose for the desired features
    # df_feat = df[["density_3d", "height", "area_3d"]]
    # # todo: call the hierarchical function
    # clustering = hierarchical.hierarchical(df_nl, df_feat, 5)


    return 0


if __name__ == "__main__":
    main()
