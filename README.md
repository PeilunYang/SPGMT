# SPGMT

Source codes for SPGMT

## Running Procedures:

1. Download and process the trajectory datasets to obtain the trajectory representations over road networks. We provide an example dataset in this project.
2. Run 'spatial_preprocess.py' to obtain the initial structural embeddings for trajectories as well as data for training, validation and test.
3. Run 'spatial_similarity_computation.py' to compute the pairwise point distances and ground truth similarities for trajectories.
4. Run 'generate_node_knn.py' to get the kNN neighbors for each node in the road network.
5. Run 'spatial_data_utils.py' to obtain the training triplets for SPGMT.
6. Run 'main.py' to train SPGMT. To test SPGMT, you can load the saved model and use the 'Spa_eval' function in the 'main.py'. The experiments on test data, long trajectories and scalibility study are all included in this function. In addition, you can modify the parameters in 'config.yaml' to run the model, and 'usePE' and 'useSI' correspond to our two variants in the ablation study.
