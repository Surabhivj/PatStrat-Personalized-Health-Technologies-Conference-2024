from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from src.data_fusion import DataFusion

rna_cell_lines = pd.read_csv("data/RNA_df.csv",index_col=0)
prot= pd.read_csv("data/PROTEIN_df.csv",index_col=0)
rna_mutation= pd.read_csv("data/RNA_MUT_df.csv",index_col=0)
protein_mutation= pd.read_csv("data/PROTEIN_MUT_df.csv",index_col=0)
cdna_mutation= pd.read_csv("data/CDNA_MUT_df.csv",index_col=0)
vaf= pd.read_csv("data/GENE_VAF_df.csv",index_col=0)

# Example data
modalities = [rna_cell_lines, prot, rna_mutation,protein_mutation,cdna_mutation,vaf]  # List of DataFrame objects
modalities_sortedindex = []
for df in modalities:
    df.sort_index(inplace=True)
    modalities_sortedindex.append(df)


top_mod = 10
k = 20
    
# Initialize and use DataFusion class
df_instance = DataFusion(modalities, top_mod, k)
[fused_net, node_feature_df,top_nodes_in_mod_net] = df_instance.data_fusion()

fused_net.to_csv("data/integrated_network.csv")
node_feature_df.to_csv("data/patient_features.csv")
