The main.py module is the only one needed to produce the results.
 
In main.py there are 3 subsections of code, each one calling for one of the 3 algorithms.
# TODO: --- --- --- --- call K-MEANS algorithm --- --- --- ---
# TODO: --- --- --- --- call DBSCAN algorithm --- --- --- ---
# TODO: --- --- --- --- call HIERARCHICAL clustering --- --- --- ---

The section for k_means is the one ready to run. To run the dbscan you need to deactivate the code for k_means
and activate it for dbscan. The same for the hierarchical. 

For all 3 algorithms, the code: 

1. first calls the function 'load_and_calculate_features' from 'calculate_features.py' module, which loads the 
point clouds and computes the features for each one of them. 
Then feeds the semantic part of the dataframe (df_nl) and the feature part (df_feat) in the clustering algorithm.

2. Inside the module of each algorithm, the plotting process as well as the validation method are called. 
