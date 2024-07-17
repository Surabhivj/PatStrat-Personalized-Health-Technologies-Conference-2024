from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from src.data_fusion import DataFusion

rna_cell_lines = pd.read_parquet("data/RNA_df.parquet")
prot= pd.read_parquet("data/PROTEIN_df.parquet")
rna_mutation= pd.read_parquet("data/RNA_MUT_df.parquet")
protein_mutation= pd.read_parquet("data/PROTEIN_MUT_df.parquet")
cdna_mutation= pd.read_parquet("data/CDNA_MUT_df.parquet")
vaf= pd.read_parquet("data/GENE_VAF_df.parquet")

# Example data
modalities = [rna_mutation, protein_mutation]  # List of DataFrame objects
modalities_sortedindex = []
n_mod = 100
n_neighbours = 10
n_steps = 3
    
# Initialize and use DataFusion class
df_instance = DataFusion(modalities, n_mod, n_neighbours, n_steps)
[fused_net, node_feature_df,top_nodes_in_mod_net] = df_instance.data_fusion()

fused_net.to_csv("data/integrated_network.csv")
node_feature_df.to_csv("data/patient_features.csv")
