
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
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


# Function to compute mutual information between two columns
def mutual_information(x, y):
    return normalized_mutual_info_score(x, y)

def sort_row(row):
    return pd.Series(sorted(row))

def compute_mutual_information(df_):
    # Normalize the data
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df_), columns=df_.columns)
    
    # Compute the correlation matrix
    mutual_info_df = df.corr()
    
    # Get the upper triangular indices
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
    pairs_df = pairs_df[pairs_df['score'] > 0.7]
    return pairs_df


def Infer_PPI(prot_expression_file,drug_response_file, gene_ids_file):

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
    gene_ids = pd.read_csv(gene_ids_file)
    id_dict = dict(zip(gene_ids['uniprot_id'], gene_ids['cosmic_gene_symbol']))
    prot_expression.rename(columns=id_dict, inplace=True)
    prot_expression = prot_expression.loc[:, prot_expression.columns.notna()]

    string_ppi_dat = pd.read_csv(os.path.join('data',"String_PPI.txt.gz"), sep= ' ')
    string_ppi_dat = string_ppi_dat[string_ppi_dat['experimental']>700].reset_index(drop=True)
    string_ppi_dat['experimental'] = string_ppi_dat['experimental'] * 0.001
    string_ppi_dat = string_ppi_dat[['protein1','protein2','experimental']]

    # To download string network metadata
    #String_info_url = "https://stringdb-static.org/download/protein.info.v11.5/9606.protein.info.v11.5.txt.gz"
    String_info_url_file = os.path.join('data',"String_protein_info.txt.gz")
    #response = wget.download(String_info_url, String_info_url_file)

    #string_ppi_dat
    string_info_dat = pd.read_csv(String_info_url_file, sep='\t')
    string_id_dict = dict(zip(string_info_dat['#string_protein_id'], string_info_dat['preferred_name']))
    string_net = string_ppi_dat.replace(string_id_dict)

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
        string_net2.reset_index(drop=True, inplace=True)
        string_net2.columns = ['source', 'target', 'score']

        # Compute mutual information and obtain DataFrame `pnet`
        pnet = compute_mutual_information(sub_prot_expression)

        # Create a set of (source, target) pairs from `string_net2` for comparison
        string_net2_pairs = set(string_net2[['source', 'target']].apply(tuple, axis=1))

        # Create a column in `pnet` indicating whether each pair is in `string_net2`
        pnet['refnet'] = pnet[['source', 'target']].apply(lambda x: 1 if tuple(x) in string_net2_pairs else 0, axis=1)

        # Filter `string_net2` to exclude pairs that are in `pnet`
        df1_pairs = set(pnet[['source', 'target']].apply(tuple, axis=1))
        df2_filtered = string_net2[~string_net2[['source', 'target']].apply(tuple, axis=1).isin(df1_pairs)]
        df2_filtered['refnet'] = 1

        # Concatenate `pnet` with the filtered `string_net2`
        protnet = pd.concat([pnet, df2_filtered], ignore_index=True)

        # Reset index of the concatenated DataFrame
        protnet.reset_index(drop=True, inplace=True)

        # Display the result
        print(protnet.head(5))
   
        protnet['type'] = 'proteomics'
        protnet.to_csv(os.path.join(protDIR, cancer + '_prot_net.tsv.gz'), sep='\t', index=True)