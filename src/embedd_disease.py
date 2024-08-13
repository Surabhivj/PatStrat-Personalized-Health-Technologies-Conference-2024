
import pandas as pd
import networkx as nx
import os
import numpy as np
from karateclub import Graph2Vec


def EMBEDD_DISEASE():

    idx = 0

    mutDIR = "results/genomics"
    protDIR = "results/proteomics"
    geneDIR = "results/transcriptomics"

    mutFILES = sorted(os.listdir(mutDIR))
    proFILES = sorted(os.listdir(protDIR))
    geneFILES = sorted(os.listdir(geneDIR))

    graphs = []
    graph_names = []

    for idx in range(len(mutFILES)):
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
            
        graphs.append(G)
        graph_names.append(mutFILES[idx].split('_')[0] )

    # Step 1: Determine the set of all nodes
    all_nodes = set()
    for graph in graphs:
        all_nodes.update(graph.nodes)

    node_index_mapping = {node: i for i, node in enumerate(sorted(all_nodes))}
    # Step 2: Add missing nodes to each graph
    for graph in graphs:
        # Find missing nodes
        missing_nodes = all_nodes - set(graph.nodes)
        
        # Add missing nodes
        for node in missing_nodes:
            graph.add_node(node)

    def relabel_graph_nodes(graph, mapping):
        return nx.relabel_nodes(graph, mapping)

    relabeled_graphs = [relabel_graph_nodes(graph, node_index_mapping) for graph in graphs]
    model = Graph2Vec()
    model.fit(relabeled_graphs)
    graph_embeddings = model.get_embedding()
    print("Embeddings shape:", graph_embeddings.shape)
    print("Embeddings:", graph_embeddings)
    graph_embeddings = np.array(graph_embeddings)
    df_embeddings = pd.DataFrame(graph_embeddings, index=graph_names)
    df_embeddings.to_csv("results/disease_embeddings.csv")

    
        