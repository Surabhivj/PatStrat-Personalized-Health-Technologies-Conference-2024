from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from src.data_fusion import DataFusion
from src.infer_grn import InferGRN,InferGRN_single
import argparse

# Set the location of the input data and the desired location of the output files

DATA_DIR = 'data/transcriptomics'
OUTPUT_DIR = '~/results/GRNs'

PRIORS_FILE_NAME = 'refnet.tsv.gz'
GOLD_STANDARD_FILE_NAME = 'refnet.tsv.gz'
TF_LIST_FILE_NAME = 'tf_names.tsv'


parser = argparse.ArgumentParser(description='Infer Gene Regulatory Networks')
parser.add_argument('--tcga_id', type=str, required=True, help='Name of the expression file')
args = parser.parse_args()
EXPRESSION_FILE_NAME = args.tcga_id + "_gene_expression.tsv.gz"


#InferGRN(DATA_DIR,PRIORS_FILE_NAME,GOLD_STANDARD_FILE_NAME,TF_LIST_FILE_NAME)

InferGRN_single(DATA_DIR,PRIORS_FILE_NAME,GOLD_STANDARD_FILE_NAME,TF_LIST_FILE_NAME,EXPRESSION_FILE_NAME)


# # Example data
# modalities = [rna_mutation, protein_mutation]  # List of DataFrame objects
# modalities_sortedindex = []
# n_mod = 100
# n_neighbours = 10
# n_steps = 3
    
# # Initialize and use DataFusion class
# df_instance = DataFusion(modalities, n_mod, n_neighbours, n_steps)
# [fused_net, node_feature_df,top_nodes_in_mod_net] = df_instance.data_fusion()

# fused_net.to_csv("results/integrated_network.csv")
# node_feature_df.to_csv("results/patient_features.csv")
