import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import zscore


def compute_mi_matrix(df):
    genes = df.columns
    mi_matrix = pd.DataFrame(index=genes, columns=genes, data=0.0)

    for gene1 in genes:
        for gene2 in genes:
            if gene1 != gene2:
                mi_matrix.loc[gene1, gene2] = mutual_info_regression(
                    df[[gene1]], df[gene2], discrete_features=True)[0]
    return mi_matrix

def compute_z_scores(mi_matrix):
    z_matrix = mi_matrix.apply(zscore, axis=1)
    return z_matrix

def compute_clr_matrix(z_matrix):
    genes = z_matrix.columns
    clr_matrix = pd.DataFrame(index=genes, columns=genes, data=0.0)

    for gene1 in genes:
        for gene2 in genes:
            if gene1 != gene2:
                z_score_1 = z_matrix.loc[gene1, gene2]
                z_score_2 = z_matrix.loc[gene2, gene1]
                clr_matrix.loc[gene1, gene2] = np.sqrt(z_score_1**2 + z_score_2**2)
    return clr_matrix
