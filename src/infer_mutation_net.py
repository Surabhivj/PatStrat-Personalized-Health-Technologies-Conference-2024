
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
import os
#import packages
import wget
from datetime import date



def Infer_MUT(mutation_file, drug_response_file):
    drug_response = pd.read_csv(drug_response_file, index_col=0)
    drug_response = drug_response[drug_response['TCGA_DESC'] != 'UNCLASSIFIED']
    # Count occurrences of each value in 'TCGA_DESC'
    value_counts = drug_response['TCGA_DESC'].value_counts()
    # Filter to keep only those values that appear more than 10,000 times
    values_to_keep = value_counts[value_counts > 10000].index
    # Filter the DataFrame to include only rows where 'TCGA_DESC' is in the values_to_keep
    drug_response = drug_response[drug_response['TCGA_DESC'].isin(values_to_keep)]
    models = drug_response.index.drop_duplicates()

    mut = pd.read_csv(mutation_file)
    mut = mut[mut['model_id'].isin(models)].reset_index(drop=True)
    mut.index = mut['model_id'].values
    mut = mut[mut['cancer_driver']==True]

    prot_mut = mut[['gene_symbol', 'protein_mutation']]
    rna_mut = mut[['gene_symbol', 'rna_mutation']]
    cdna_mut = mut[['gene_symbol', 'cdna_mutation']]

    # Assuming drug_response, prot_mut, cdna_mut, and rna_mut are already defined
    cancers = drug_response['TCGA_DESC'].unique()

    # Define the directory path
    mutDIR = 'results/genomics/'

    # Create the directory path if it doesn't exist
    os.makedirs(mutDIR, exist_ok=True)

    for cancer in cancers:
        all_mut = pd.DataFrame(columns=['gene_symbol', 'mutation'])
        for mutation_table in [prot_mut, cdna_mut, rna_mut]:
            mutation_table = mutation_table[(mutation_table.iloc[:, 1] != '-') & 
                                            (mutation_table.iloc[:, 1] != 'p.?') & 
                                            (mutation_table.iloc[:, 1] != 'r.?') &
                                            (mutation_table.iloc[:, 1] != 'c.?')].reset_index(drop=True)
            mutation_table.columns = ['gene_symbol', 'mutation']
            all_mut = pd.concat([all_mut, mutation_table], ignore_index=True)
        
        # Save to CSV
        output_file = os.path.join(mutDIR, f'{cancer}_mutations.tsv.gz')
        all_mut.to_csv(output_file, sep='\t', index=False)
    