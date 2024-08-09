
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
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#import packages
import pandas as pd
import wget
from datetime import date
import gzip
import pandas as pd
from itertools import combinations




idx = 0

mutDIR = "results/genomics"
protDIR = "results/proteomics"
geneDIR = "results/transcriptomics_old"

mutFILES = sorted(os.listdir(mutDIR))
proFILES = sorted(os.listdir(protDIR))
geneFILES = sorted(os.listdir(geneDIR))

mnet = pd.read_csv(os.path.join(mutDIR, mutFILES[idx]), sep = "\t")
pnet = pd.read_csv(os.path.join(protDIR,proFILES[idx]), sep = "\t", index_col=0)
gnet = pd.read_csv(os.path.join(geneDIR, geneFILES[idx],'final', 'network.tsv.gz'), sep = "\t")

gnet = gnet[['target','regulator','combined_confidences','gold_standard']]
gnet.columns = ['target','source','score_transcriptomics','known_regulatory_link']
gnet['rna_edgetype'] = 'transcriptomics'
pnet.columns = ['source',	'target',	'score_proteomics',	'known_PPI',	'prot_edgetype']
mnet.columns = ['source', 'target']
mnet['mut_edgetype'] = 'genomics'

net = pd.merge(pd.merge(gnet, pnet, on=['source', 'target'], how='outer'), mnet, on=['source', 'target'], how='outer')

net.to_csv(os.path.join("results", 'disease_net', mutFILES[idx].split('_')[0] + '.csv'))

# Initialize a directed graph
G = nx.DiGraph()

# Iterate over the DataFrame rows and add edges to the graph
for index, row in net.iterrows():
    # Add edge with attributes
    G.add_edge(row['source'], row['target'], 
               score_transcriptomics=row['score_transcriptomics'],
               known_regulatory_link=row['known_regulatory_link'],
               rna_edgetype=row['rna_edgetype'],
               score_proteomics=row['score_proteomics'],
               known_PPI=row['known_PPI'],
               prot_edgetype=row['prot_edgetype'],
               mut_edgetype=row['mut_edgetype'])
    
# Generate Node2Vec model
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)

# Fit the model and generate embeddings
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Get the embeddings for each node in the graph
embeddings = {node: model.wv[node] for node in G.nodes()}

# Display embeddings for each node
for node, embedding in embeddings.items():
    print(f"Node: {node}\nEmbedding: {embedding}\n")
