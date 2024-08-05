from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
import os
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
#import packages
import wget
from datetime import date
import gzip
from sklearn.preprocessing import MinMaxScaler
from src.snf import snf

import warnings
warnings.filterwarnings("ignore")


# Function to compute mutual information between two columns
def mutual_information(x, y):
    return normalized_mutual_info_score(x, y)

# Compute mutual information for each pair of proteins
def compute_mutual_information(df):
    # Initialize a DataFrame to store mutual information scores
    mutual_info_df = pd.DataFrame(index=df.columns, columns=df.columns)

    # Compute mutual information for each pair of columns
    for col1 in df.columns:
        for col2 in df.columns:
            mutual_info_df.loc[col1, col2] = mutual_information(df[col1], df[col2])
    
    # Create a list to store pairs with mutual information > 0.7
    pairs = []
    for col1 in mutual_info_df.columns:
        for col2 in mutual_info_df.columns:
            if col1 != col2:
                mi_score = mutual_info_df.loc[col1, col2]
                if mi_score > 0.7:
                    pairs.append((col1, col2, mi_score))

    # Create a DataFrame for the pairwise scores
    pairwise_df = pd.DataFrame(pairs, columns=['protein1', 'protein2', 'Mutual Information'])
    pairwise_df = pairwise_df[['protein1', 'protein2']]
    
    return pairwise_df

# Function to generate unique sorted pairs, excluding pairs with the same ID
def generate_unique_pairs(data):
    seen_pairs = set()
    for gene, group in data.groupby('gene_symbol'):
        indices = group.index.tolist()
        for pair in combinations(indices, 2):
            if pair[0] != pair[1]:  # Ensure the pair does not contain the same ID twice
                sorted_pair = tuple(sorted(pair))
                if sorted_pair not in seen_pairs:
                    seen_pairs.add(sorted_pair)
                    yield sorted_pair    


def Infer_PSN(mutation_file,prot_expression_file,gene_expression_file,drug_response_file, n_neighbors = 20):
    prot_expression = pd.read_csv(prot_expression_file, index_col=0)
    gene_expression = pd.read_csv(gene_expression_file, index_col=0)
    mut = pd.read_csv(mutation_file)
    
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
    mut = mut[mut['model_id'].isin(models)].reset_index(drop=True)
    mut.index = mut['model_id'].values
    data = mut[['gene_symbol']]
    data.sort_values('gene_symbol')

    prot_expression = prot_expression.T
    gene_expression = gene_expression.T

    print("Computing Pnet...")

    pnet = compute_mutual_information(prot_expression)
    pnet.to_csv(os.path.join("result", "proteomics", "PSN_prot_net.tsv.gz"), sep='\t', index=True)

    print("Computing Gnet...")
    gnet = compute_mutual_information(gene_expression)
    gnet.to_csv(os.path.join("result", "transcriptomics", "PSN_rna_net.tsv.gz"), sep='\t', index=True)

    print("Computing Mnet...")

    unique_pairs = list(generate_unique_pairs(data))
    mnet = pd.DataFrame(unique_pairs, columns=['patient1', 'patient2'])
    mnet.to_csv(os.path.join("result", "genomics", "PSN_mut_net.tsv.gz"), sep='\t', index=True)

    print("Computing integrated PSN...")
    
    fused_net = snf([gnet, pnet, mnet], K=n_neighbors)
    fused_net.to_csv("results/integrated_PSN.csv")


