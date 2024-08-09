from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import networkx as nx
import os
from scipy.stats import zscore
#import packages
import wget
from datetime import date
import gzip
from sklearn.preprocessing import MinMaxScaler
from src.snf import snf
from itertools import combinations
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")


def sort_row(row):
    return pd.Series(sorted([row['source'], row['target']]))

def compute_mutual_information(df_, varfile, top = 200):
    # Normalize the data
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df_), columns=df_.columns, index=df_.index)

    # Calculate variance of each feature
    variances = df.var(axis=0)
    
    # Select top_n most variable genes
    top_genes = variances.nlargest(top).index
    bottom_genes = variances.nsmallest(top).index
    
    # Write top and bottom variable genes to one file
    with open(varfile, 'w') as f:
        f.write("Top Variable Genes:\n")
        for gene in top_genes:
            f.write(f"{gene}\n")
        
        f.write("\nBottom Variable Genes:\n")
        for gene in bottom_genes:
            f.write(f"{gene}\n")
    
    # Select top and bottom genes from the DataFrame
    df_top = df[top_genes]
    df_bottom = df[bottom_genes]
    
    # Combine top and bottom genes
    df = pd.concat([df_top, df_bottom], axis=1)

    # Compute the correlation matrix
    mutual_info_df = df.corr(method='spearman')
    upper_tri_indices = np.triu_indices(mutual_info_df.shape[0], k=1)
    
    # Extract the pairs and values
    pairs = []
    values = []
    for i, j in zip(upper_tri_indices[0], upper_tri_indices[1]):
        pairs.append((df.columns[i], df.columns[j]))
        values.append(mutual_info_df.iloc[i, j])

    # Create a DataFrame with pairs and their values
    pairs_df = pd.DataFrame({
        'source': [p[0] for p in pairs],
        'target': [p[1] for p in pairs],
        'score': values
    })
    pairs_df = pairs_df[pairs_df['score'] > 0.8]
    return pairs_df


def replace_and_generate_matrix(gnet, string_to_number):
    
    # Step 1: Replace protein names with numbers
    gnet.iloc[:, 0] = gnet.iloc[:, 0].map(string_to_number)
    gnet.iloc[:, 1] = gnet.iloc[:, 1].map(string_to_number)
    
    # Step 2: Get the list of unique numerical IDs from the dictionary
    unique_proteins = list(string_to_number.values())
    
    # Step 3: Initialize the matrix with zeros
    matrix = pd.DataFrame(0, index=unique_proteins, columns=unique_proteins)
    
    # Step 4: Fill the matrix
    for _, row in gnet.iterrows():
        source, target = row.iloc[0], row.iloc[1]
        if source in matrix.index and target in matrix.columns:
            matrix.loc[source, target] = 1  # Set a value of 1 for the presence of an interaction
            matrix.loc[target, source] = 1  # Ensure the matrix is symmetric for undirected interactions
    return matrix


# Function to generate unique sorted pairs, excluding pairs with the same ID
def generate_unique_pairs(data):
    seen_pairs = defaultdict(int)
    for gene, group in data.groupby('gene_symbol'):
        indices = group.index.tolist()
        for pair in combinations(indices, 2):
            if pair[0] != pair[1]:  # Ensure the pair does not contain the same ID twice
                sorted_pair = tuple(sorted(pair))
                seen_pairs[sorted_pair] += 1

    # Convert the dictionary to a sorted list of tuples
    sorted_pairs = sorted(seen_pairs.items())
    return sorted_pairs

def convert_to_dataframe(pairs_counts):
    data = {
        'source': [pair[0][0] for pair in pairs_counts],
        'target': [pair[0][1] for pair in pairs_counts],
        'score': [pair[1] for pair in pairs_counts]
    }
    df = pd.DataFrame(data)
    scaler = MinMaxScaler()
    df[['score']] = scaler.fit_transform(df[['score']])  
    df = df[df['score'] > 0.2]
    return df

def Infer_PSN(mutation_file,prot_expression_file,gene_expression_file,drug_response_file, n_neighbors = 20):
    prot_expression = pd.read_csv(prot_expression_file, index_col=0)
    gene_expression = pd.read_csv(gene_expression_file, index_col=0)
    mut = pd.read_csv(mutation_file)
    mut = mut[mut['cancer_driver']==True]
    
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
    mut = mut[mut['model_id'].isin(models)].reset_index(drop=False)
    mut.index = mut['model_id'].values
    data = mut[['gene_symbol']]
    data.sort_values('gene_symbol')

    prot_expression = prot_expression.T
    gene_expression = gene_expression.T

    print("Computing Pnet...")
    pnet = compute_mutual_information(prot_expression,varfile=os.path.join("results", "PSN_prot_topfeatures.txt"), top=200)
    pnet['type'] = 'proteomics'
    pnet.to_csv(os.path.join("results", "PSN_prot_net.csv"), index=False)

    print("Computing Gnet...")
    gnet = compute_mutual_information(gene_expression, varfile=os.path.join("results", "PSN_rna_topfeastures.txt"), top=2000)
    gnet['type'] = 'transcriptomics'
    gnet.to_csv(os.path.join("results", "PSN_rna_net.csv"), index=False)

    print("Computing Mnet...")

    # Generate unique pairs and their counts
    pairs_counts = generate_unique_pairs(data)

    # Convert pairs and counts to a DataFrame
    mnet = convert_to_dataframe(pairs_counts)
    #mnet = mnet.apply(sort_row, axis=1)
    #mnet = mnet.drop_duplicates(subset=['source', 'target'])
    mnet['type'] = 'genomics'
    mnet.to_csv(os.path.join("results", "PSN_mut_net.csv"), index=False)

    print("Computing integrated PSN...")

    string_to_number = {s: i for i, s in enumerate(set(drug_response.index))}
    gnet = replace_and_generate_matrix(gnet, string_to_number)
    pnet = replace_and_generate_matrix(pnet, string_to_number)
    mnet = replace_and_generate_matrix(mnet, string_to_number)
    
    fused_net = snf([gnet, pnet, mnet], K=n_neighbors)
    number_to_string = {float(i): s for s, i in string_to_number.items()}

    # Get the non-zero indices of `fused_net`
    nonzero_indices = np.nonzero(fused_net)

    # Create a DataFrame from the non-zero values
    net = pd.DataFrame(
        np.vstack((nonzero_indices[0], nonzero_indices[1], fused_net[nonzero_indices])).T,
        columns=['source', 'target', 'score']
    )
    net['source'].replace(number_to_string, inplace=True)
    net['target'].replace(number_to_string, inplace=True)
    net['type'] = 'integrated'

    net = net[net['source'] != net['target']]
    scaler = MinMaxScaler()
    net[['score']] = scaler.fit_transform(net[['score']])  
    net = net[net['score'] > 0.03]
    
    # Create a list of all unique samples
    samples = net['source'].tolist() + net['target'].tolist()

    drug_response['patient'] = drug_response.index.values
    node_info = drug_response[['patient','TCGA_DESC']].drop_duplicates()
    node_info.reset_index(inplace=True, drop=True)
    node_info = node_info[node_info['patient'].isin(samples)]
    node_info.reset_index(inplace=True, drop=True)
    #node_info.to_csv("results/integrated_PSN_nodes.csv", index=False)

    int_net = net.drop_duplicates()
    prot_net = pd.read_csv(os.path.join("results", "PSN_prot_net.csv")).drop_duplicates()
    rna_net = pd.read_csv(os.path.join("results", "PSN_rna_net.csv")).drop_duplicates()
    mut_net = pd.read_csv(os.path.join("results", "PSN_mut_net.csv")).drop_duplicates()

    merged_df1 = pd.merge(int_net, prot_net, on=['source', 'target'], suffixes=('_int', '_prot'), how='left')
    merged_df2 = pd.merge(merged_df1, rna_net, on=['source', 'target'],how='left')
    merged_df = pd.merge(merged_df2, mut_net, on=['source', 'target'], how='left', suffixes=('_rna', '_mut'))
    # Merge for the source labels
    source_labels = node_info.rename(columns={'patient': 'source', 'TCGA_DESC': 'source_Label'})
    merged_with_source = pd.merge(merged_df, source_labels, on='source', how='left')

    # Merge for the target labels
    target_labels = node_info.rename(columns={'patient': 'target', 'TCGA_DESC': 'target_Label'})
    final_merged_df = pd.merge(merged_with_source, target_labels, on='target', how='left')
    final_merged_df.to_csv("results/integrated_PSN.csv", index=False)

    node_info = node_info.rename(columns={'patient': 'Id', 'TCGA_DESC': 'Label'})
    node_info.to_csv("results/integrated_PSN_node.csv", index=False)

    

