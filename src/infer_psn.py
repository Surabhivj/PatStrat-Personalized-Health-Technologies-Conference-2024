from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
import os
from sklearn.metrics import mutual_info_score
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
#import packages
import wget
from datetime import date
import gzip
from sklearn.preprocessing import MinMaxScaler


import warnings
warnings.filterwarnings("ignore")

# Function to compute mutual information between two columns
def mutual_information(x, y):
    return mutual_info_score(x, y)

# Compute mutual information for each pair of proteins
def compute_mutual_information(df):
    mutual_info_df = pd.DataFrame(index=df.columns, columns=df.columns)
    for col1 in df.columns:
        for col2 in df.columns:
            mutual_info_df.loc[col1, col2] = mutual_information(df[col1], df[col2])
    
    # Convert mutual information matrix to pairwise format
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the data
    mutual_df = scaler.fit_transform(mutual_info_df)

    # Convert the result back to a DataFrame
    mutual_info_df = pd.DataFrame(mutual_df, columns=df.columns)
    return mutual_info_df



def Infer_PSN(mutation_file,prot_expression_file,gene_expression_file,drug_response_file):
    mut = pd.read_csv(mutation_file)
    prot_expression = pd.read_csv(prot_expression_file, index_col=0)
    gene_expression = pd.read_csv(gene_expression_file, index_col=0)

    drug_response = pd.read_csv(drug_response_file, index_col=0)
    drug_response = drug_response[drug_response['TCGA_DESC'] != 'UNCLASSIFIED']
    # Count occurrences of each value in 'TCGA_DESC'
    value_counts = drug_response['TCGA_DESC'].value_counts()
    # Filter to keep only those values that appear more than 10,000 times
    values_to_keep = value_counts[value_counts > 10000].index
    # Filter the DataFrame to include only rows where 'TCGA_DESC' is in the values_to_keep
    drug_response = drug_response[drug_response['TCGA_DESC'].isin(values_to_keep)]
    models = drug_response.index.drop_duplicates()

    prot_expression = prot_expression[prot_expression.index.isin(models)]
    gene_expression = gene_expression[gene_expression.index.isin(models)]

    prot_expression = prot_expression.T
    gene_expression = gene_expression.T

    print("Computing Pnet...")

    pnet = compute_mutual_information(prot_expression)

    print("Computing Gnet...")
    gnet = compute_mutual_information(gene_expression)

    






# Example data
modalities = [rna_mutation, protein_mutation]  # List of DataFrame objects
modalities_sortedindex = []
n_mod = 100
n_neighbours = 10
n_steps = 3
    
# Initialize and use DataFusion class
df_instance = DataFusion(modalities, n_mod, n_neighbours, n_steps)
[fused_net, node_feature_df,top_nodes_in_mod_net] = df_instance.data_fusion()

fused_net.to_csv("results/integrated_network.csv")
node_feature_df.to_csv("results/patient_features.csv")
