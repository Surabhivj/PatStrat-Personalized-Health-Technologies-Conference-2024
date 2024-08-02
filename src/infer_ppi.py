
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
import os
from scipy.stats import zscore
#import packages
import wget
from datetime import date
import gzip
from sklearn.metrics import normalized_mutual_info_score


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


def Infer_PPI(prot_expression_file,drug_response_file):

    drug_response = pd.read_csv(drug_response_file, index_col=0)
    drug_response = drug_response[drug_response['TCGA_DESC'] != 'UNCLASSIFIED']
    # Count occurrences of each value in 'TCGA_DESC'
    value_counts = drug_response['TCGA_DESC'].value_counts()
    # Filter to keep only those values that appear more than 10,000 times
    values_to_keep = value_counts[value_counts > 10000].index
    # Filter the DataFrame to include only rows where 'TCGA_DESC' is in the values_to_keep
    drug_response = drug_response[drug_response['TCGA_DESC'].isin(values_to_keep)]
    models = drug_response.index.drop_duplicates()

    prot_expression = pd.read_csv(prot_expression_file, index_col=0)
    prot_expression = prot_expression[prot_expression.index.isin(models)]
    prot_expression = prot_expression.add_prefix('P_')

    string_ppi_dat = pd.read_csv(os.path.join('data',"String_PPI.txt.gz"), sep= ' ')
    string_ppi_dat = string_ppi_dat[string_ppi_dat['experimental']>700].reset_index(drop=True)
    string_ppi_dat = string_ppi_dat[['protein1','protein2']]
    string_ppi_dat

    # To download string network metadata
    #String_info_url = "https://stringdb-static.org/download/protein.info.v11.5/9606.protein.info.v11.5.txt.gz"
    String_info_url_file = os.path.join('data',"String_protein_info.txt.gz")
    #response = wget.download(String_info_url, String_info_url_file)

    #string_ppi_dat
    string_info_dat = pd.read_csv(String_info_url_file, sep='\t')
    string_id_dict = dict(zip(string_info_dat['#string_protein_id'], string_info_dat['preferred_name']))
    string_net = string_ppi_dat.replace(string_id_dict)

    # Define the prefix
    prefix = 'P_'
    # Add the prefix to all elements in the DataFrame
    string_net = string_net.applymap(lambda x: prefix + x)
    cancers = drug_response['TCGA_DESC'].unique()
    # Define the directory path
    protDIR = 'results/proteomics/'

    # Create the directory path if it doesn't exist
    os.makedirs(protDIR, exist_ok=True)
    for cancer in cancers:
        print("Cancer:" + cancer)
        cancer_samples = list(drug_response[drug_response['TCGA_DESC'] == cancer].index)
        sub_prot_expression = prot_expression[prot_expression.index.isin(cancer_samples)]
        string_net1 = string_net[string_net['protein1'].isin(sub_prot_expression.columns)]
        string_net2 = string_net1[string_net1['protein2'].isin(sub_prot_expression.columns)]
        #sub_prot_expression = sub_prot_expression.iloc[:10, :50]
        print("Computing MI matrix...")
        pnet = compute_mutual_information(sub_prot_expression)
        print(pnet)
        # Create pairwise DataFrame
        protnet = pd.concat([pnet, string_net2], ignore_index=True)
        protnet.to_csv(os.path.join(protDIR, cancer + '_prot_net.tsv.gz'), sep='\t', index=True)