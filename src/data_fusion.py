import numpy as np
import networkx as nx
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import pairwise_distances
from src.snf import snf

class DataFusion:
    def __init__(self, modalities, n_mod, n_neighbours, n_steps):
        self.modalities = modalities
        self.top_mod = n_mod
        self.k = n_neighbours
        self.n_steps = n_steps
        self.patient_nets = []
        self.top_nodes_in_mod_net = []
        self.node_feature_df = []

    def data_fusion(self):
        for mod in self.modalities:
            print(mod.shape)
            # Handle missing values
            if mod.isna().sum().sum() > 0:
                imputer = IterativeImputer()
                filled_data = imputer.fit_transform(mod)
            else:
                filled_data = mod
            
            # Perform SVD
            U, S, VT = np.linalg.svd(filled_data, full_matrices=False)
            
            # Compute embeddings
            patient_emb = U * S
            mod_embedding = VT.T * S
            
            # Compute patient network
            patient_net = self.compute_rw_matrix(patient_emb, n_steps=self.n_steps)
            self.patient_nets.append(patient_net)
            
            # Compute modality network
            mod_net = np.corrcoef(mod_embedding)
            G = nx.from_numpy_array(mod_net)
            mapping = {i: label for i, label in enumerate(mod.columns)}
            G = nx.relabel_nodes(G, mapping)
            pr = nx.pagerank(G, alpha=0.85)
            sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
            topm = [node for node, _ in sorted_pr[:self.top_mod]]
            self.top_nodes_in_mod_net.append(topm)
            
            # Subset modality features
            subset_mod = mod[mod.columns.intersection(self.top_nodes_in_mod_net[-1])]
            self.node_feature_df.append(subset_mod)
        
        # Perform Similarity Network Fusion
        fused_net = snf(self.patient_nets, K=self.k)
        
        return [fused_net, self.node_feature_df, self.top_nodes_in_mod_net]

    @staticmethod
    def compute_rw_matrix(A, n_steps):
        D = np.diag(np.sum(A, axis=1))
        P = np.dot(np.linalg.inv(D), A)
        W = np.linalg.matrix_power(P, n_steps)
        return W

