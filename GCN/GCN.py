import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Function to encode class labels as integer values
def encode_labels(labels):
    # Create a mapping from class labels to one-hot encoded vectors
    classes = set(labels)
    class_map = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_encoded = np.array(list(map(class_map.get, labels)), dtype=np.int32)
    return labels_encoded.argmax(axis=1)  # Convert one-hot encoded vectors to integer labels for PyTorch compatibility

# Function to normalize the feature matrix row-wise
def normalize_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    features = features * r_inv[:, None]
    return features

# Function to normalize the adjacency matrix for GCN
def normalize_adj(adj):
    # Add self-loops to ensure every node can pass its own features forward
    adj += torch.eye(adj.shape[0])
    rowsum = adj.sum(1).pow(-0.5)
    rowsum[rowsum == float('inf')] = 0
    return adj * rowsum[:, None] * rowsum[None, :]

# Function to load and preprocess the CORA dataset content
def load_cora_content(file_path):
    data = np.genfromtxt(file_path, dtype=np.dtype(str))
    features = np.array(data[:, 1:-1], dtype=np.float32)
    features = normalize_features(features)
    labels = encode_labels(data[:, -1])
    paper_id = data[:, 0]
    id_map = {j: i for i, j in enumerate(paper_id)}
    return torch.tensor(features), torch.tensor(labels), id_map

# Function to load the CORA citation data and construct the adjacency matrix
def load_cora_cites(train_path, test_path, id_map):
    cites_train = np.genfromtxt(train_path, dtype=np.dtype(str))
    cites_test = np.genfromtxt(test_path, dtype=np.dtype(str))
    cites_combined = np.concatenate((cites_train, cites_test), axis=0)

    # Convert paper IDs in citation data to indices using the id_map
    train_nodes = np.array(list(map(id_map.get, cites_train[:, -1])), dtype=np.int32)
    test_nodes = np.array(list(map(id_map.get, cites_test[:, -1])), dtype=np.int32)

    # Create symmetric adjacency matrix
    edges = np.array(list(map(id_map.get, cites_combined.flatten())), dtype=np.int32).reshape(cites_combined.shape)
    adj = np.zeros((len(id_map), len(id_map)), dtype=np.float32)
    adj[edges[:, 0], edges[:, 1]] = 1
    adj[edges[:, 1], edges[:, 0]] = 1

    return torch.tensor(adj), torch.tensor(train_nodes), torch.tensor(test_nodes)

# Graph Convolutional Network (GCN) layer definition
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        return output

# GCN model comprising two GCN layers
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(nfeat, nhid)
        self.gc2 = GCNLayer(nhid, nclass)
        self.dropout = dropout
    
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

# Function to train the GCN model
def train(model, features, adj, labels, idx_train, optimizer, num_epochs=200):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(features, adj)
        preds = output.max(1)[1].type_as(labels)
        train_preds = preds[idx_train]
        train_labels = labels[idx_train]
        acc = accuracy_score(train_labels, train_preds)

        loss = F.nll_loss(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}')
        if epoch == (num_epochs - 1):
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}')
            
# Function to evaluate the GCN model
def evaluate(model, features, adj, labels, idx_list):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        preds = output.max(1)[1].type_as(labels)
        
        pred_labels = preds[idx_list]
        true_labels = labels[idx_list]

        precision = precision_score(true_labels, pred_labels, average='macro')
        recall = recall_score(true_labels, pred_labels, average='macro')
        f1 = f1_score(true_labels, pred_labels, average='macro')
        accuracy = accuracy_score(true_labels, pred_labels)

    return accuracy, precision, recall, f1

# Main block to load data, initialize the model, and perform training and evaluation
if __name__ == '__main__':
    # Paths to dataset files
    content_path = '../dataset/cora.content'
    cites_train_path = '../dataset/cora_train.cites'
    cites_test_path = '../dataset/cora_test.cites'

    print('>> Loading the citation graph')
    features, labels, id_map = load_cora_content(content_path)
    adj, idx_train, idx_test = load_cora_cites(cites_train_path, cites_test_path, id_map)
    adj = normalize_adj(adj)
    labels = torch.LongTensor(labels)

    print('>> Initializing the model parameters')
    nfeat = features.shape[1]
    nhid = 16 # no. of hidden units
    nclass = labels.max().item() + 1
    dropout = 0.5

    print('>> Initializing the model')
    model = GCN(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=dropout)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Train the model
    print('>> Training')
    train(model, features, adj, labels, idx_train, optimizer, num_epochs=200)

    # Evaluate the model on the train and test set
    print('>> Evaluating')
    train_acc, train_prec, train_rec, train_f1 = evaluate(model, features, adj, labels, idx_train)
    test_acc, test_prec, test_rec, test_f1 = evaluate(model, features, adj, labels, idx_test)

    print(f'Train Metrics:\nAccuracy : {train_acc:.2f}, Precision : {train_prec:.2f}, Recall : {train_rec:.2f}, F1-Score : {train_f1:.2f}\n')
    print(f'Test Metrics:\nAccuracy : {test_acc:.2f}, Precision : {test_prec:.2f}, Recall : {test_rec:.2f}, F1-Score : {test_f1:.2f}\n')

    # Write evaluation metrics to a file
    with open('gcn_metrics.txt', 'w', encoding='utf-8') as file:
        file.write(f'Train Metrics:\nAccuracy : {train_acc:.2f}, Precision : {train_prec:.2f}, Recall : {train_rec:.2f}, F1-Score : {train_f1:.2f}\n')
        file.write(f'Test Metrics:\nAccuracy : {test_acc:.2f}, Precision : {test_prec:.2f}, Recall : {test_rec:.2f}, F1-Score : {test_f1:.2f}\n')
