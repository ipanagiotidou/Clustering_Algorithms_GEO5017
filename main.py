### ------ Modules -------
import DBSCAN
import k_means
import hierarchical
import calculate_features

def main():

    ### TODO: IMPORTANT NOTE --> ACTIVATE EACH SECTION AT A TIME AND RUN.

    # # TODO: --- --- --- --- call K-MEANS algorithm --- --- --- ---
    # # calculate features
    # df = calculate_features.load_and_calculate_features()
    # df_nl = df[['bname', 'label']]
    # # choose for the desired features
    # df_feat = df[["density_3d", "height", "area_3d"]] # chosen
    # # call k_means function
    # k_means.k_means(df_nl, df_feat)


    # # TODO: --- --- --- --- call DBSCAN algorithm --- --- --- ---
    # # calculate features
    # df = calculate_features.load_and_calculate_features()
    # df_nl = df[['bname', 'label']]
    # # choose for the desired features
    # df_feat = df[["density_3d", "height", "area_3d"]]
    # # Set the parameters for dbscan
    # minPts = 10
    # epsilon = 5
    # # call the dbscan function
    # DBSCAN.dbscan(df_nl, df_feat, minPts, epsilon)


    # # TODO: --- --- --- --- call HIERARCHICAL clustering --- --- --- ---
    # # calculate features
    # df = calculate_features.load_and_calculate_features()
    # df_nl = df[['bname', 'label']]
    # # choose for the desired features
    # df_feat = df[["density_3d", "height", "area_3d"]]
    # # call the hierarchical function
    # hierarchical.hierarchical(df_nl, df_feat, 5)


    return 0


if __name__ == "__main__":
    main()
