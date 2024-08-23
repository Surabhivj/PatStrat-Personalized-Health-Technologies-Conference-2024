import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.utils import from_networkx
from torch_geometric.nn import SAGEConv
from torch.nn import CrossEntropyLoss
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np


def extract_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
    return embeddings

def save_embeddings(embeddings, node_ids,output_path):
    # Convert embeddings to a NumPy array
    embeddings_np = embeddings.numpy()
    df_embeddings = pd.DataFrame(embeddings_np, index=node_ids)
    df_embeddings.to_csv(output_path)

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def load_data(disease_embedd_Path, PSNpath):
    disease_embedd = pd.read_csv(disease_embedd_Path, index_col=0)
    PSN = pd.read_csv(PSNpath)
    
    patient_info = pd.DataFrame({
        'patients': list(PSN['source']) + list(PSN['target']),
        'cancer': list(PSN['source_Label']) + list(PSN['target_Label'])
    }).drop_duplicates().reset_index(drop=True)
    return disease_embedd, PSN, patient_info

def preprocess_data(disease_embedd, PSN, patient_info):
    # Initialize the label encoder
    label_encoder = LabelEncoder()
    # Fit and transform the 'cancer' column
    patient_info['cancer_encoded'] = label_encoder.fit_transform(patient_info['cancer'])
    cancer_to_encoded = dict(zip(patient_info['cancer'], patient_info['cancer_encoded']))
    
    G = nx.from_pandas_edgelist(
        PSN,
        source='source',
        target='target',
        edge_attr=['score_int', 'type_int', 'score_prot', 'type_prot',
                   'score_rna', 'type_rna', 'score_mut', 'type_mut', 
                   'source_Label', 'target_Label'])
    
    for index, crow in disease_embedd.iterrows():
        cancer = index
        encoded_cancer = cancer_to_encoded.get(index, -1)
        patient_selc = list(patient_info[patient_info['cancer'] == cancer]['patients'])
        for node in patient_selc:
            if node in G.nodes:
                G.nodes[node]['feature_vector'] = crow.values
                G.nodes[node]['label'] = encoded_cancer
    data = from_networkx(G)
    data.x = torch.tensor([G.nodes[node]['feature_vector'] for node in G.nodes], dtype=torch.float)
    labels = [G.nodes[node].get('label', -1) for node in G.nodes]
    data.y = torch.tensor(labels, dtype=torch.long)
    data.train_mask = torch.rand(data.num_nodes) < 0.8
    data.val_mask = ~data.train_mask
    return data, len(label_encoder.classes_), list(G.nodes)

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = self.relu(conv(x, edge_index))
        x = self.convs[-1](x, edge_index)
        return x

def train(model, optimizer, criterion, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
    return acc

def EMBEDD_PATIENTS(disease_embedd_Path, PSNpath):
    disease_embedd, PSN, patient_info = load_data(disease_embedd_Path, PSNpath)
    data, num_classes,node_ids = preprocess_data(disease_embedd, PSN, patient_info)
    model = GraphSAGE(in_channels=data.num_node_features, hidden_channels=256, out_channels=128, num_layers=3)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = CrossEntropyLoss()
    epochs = []
    train_losses = []
    val_accuracies = []
    for epoch in range(200):
        loss = train(model, optimizer, criterion, data)
        train_acc = test(model, data, data.train_mask)
        val_acc = test(model, data, data.val_mask)
        epochs.append(epoch)
        train_losses.append(loss)
        val_accuracies.append(val_acc)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    # Extract and save embeddings
    embeddings = extract_embeddings(model, data)
    save_embeddings(embeddings,node_ids, "results/PatStrat_emb.csv")
    save_model(model, "results/PatStrat_model.pth")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/training_loss_val_acc.png")
