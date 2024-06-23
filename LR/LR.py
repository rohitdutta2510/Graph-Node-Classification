# Import necessary libraries
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Define a function to load the citation graph from a given training and testing .cites files
def get_citation_graph(train_file_path, test_file_path):
    # Initialize sets to hold unique nodes from the train and test datasets
    train_nodes = set()
    test_nodes = set()
    # Initialize an empty graph
    citation_graph = nx.Graph()

    # Read the training file and add edges to the citation graph
    with open(train_file_path, 'r') as file:
        for line in file:
            # Each line contains a citation link between a source and a target paper
            target, source = map(int, line.strip().split())
            # Add this citation link as an edge in the graph
            citation_graph.add_edge(source, target)
            # Record the source node in the train_nodes set
            train_nodes.add(source)

    # Repeat the process for the test file
    with open(test_file_path, 'r') as file:
        for line in file:
            target, source = map(int, line.strip().split())
            citation_graph.add_edge(source, target)
            test_nodes.add(source)

    return citation_graph, train_nodes, test_nodes

# Function to load node labels from a .content file
def get_node_label(file_path):
    node_label_map = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Each line represents a node and its label
            parts = line.strip().split()
            # Map node ID to its label
            node_label_map[int(parts[0].strip())] = parts[-1].strip()

    return node_label_map

# Function to calculate transition probabilities for random walks
def get_probs(graph, p=1, q=1):
    transition_probs = {}
    for src_node in graph.nodes():
        transition_probs[src_node] = {}
        for cur_node in graph.neighbors(src_node):
            prob_list = []
            for target_node in graph.neighbors(cur_node):
                # Calculate transition probabilities based on node relations
                if target_node == src_node:
                    # trans_prob = graph[cur_node][target_node].get('weight', 1) * (1/p)
                    prob = 1/p
                elif target_node in list(graph[src_node]):
                    # trans_prob = graph[cur_node][target_node].get('weight', 1)
                    prob = 1
                else:
                    # trans_prob = graph[cur_node][target_node].get('weight', 1) * (1/q)
                    prob = 1/q

                prob_list.append(prob)

            # Normalize probabilities so they sum to 1
            transition_probs[src_node][cur_node] = prob_list/np.sum(prob_list)
    
    return transition_probs

# Function to generate random walks based on transition probabilities
def gen_rand_walk(graph, prob_dict, max_walk=5, walk_len=10):
    walk_list = []
    for src_node in graph.nodes():
        for i in range(max_walk):
            walk = [src_node]
            neighbors = list(graph[src_node])

            if len(neighbors) == 0:
                break

            # Choose the next node based on transition probabilities
            next_node = np.random.choice(neighbors)
            walk.append(next_node)

            for j in range(walk_len - 2):
                neighbors = list(graph[walk[-1]])
                if len(neighbors) == 0:
                    break

                transition_prob = prob_dict[walk[-2]][walk[-1]]
                next_node = np.random.choice(neighbors, p=transition_prob)
                walk.append(next_node)

            walk_list.append(walk)

    return walk_list

# Function to generate node embeddings from random walks
def generate_embeddings(walk_list, win_size, embedding_size):
    # Train a Word2Vec model on the random walks
    model = Word2Vec(sentences=walk_list, window=win_size, vector_size=embedding_size)
    return model.wv

# Function to prepare training and testing datasets
def prepare_train_test_data(embeddings, train_nodes, test_nodes, node_label_map):
    X_train, y_train, X_test, y_test = [], [], [], []
    for node, label in node_label_map.items():
        if node in train_nodes:
            X_train.append(embeddings[node])
            y_train.append(label)
        elif node in test_nodes:
            X_test.append(embeddings[node])
            y_test.append(label)

    # Encode the labels into integers
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    # File paths for the dataset
    train_path = '../dataset/cora_train.cites'
    test_path = '../dataset/cora_test.cites'
    file_path = '../dataset/cora.content'

    # Load the citation network and node labels
    print('>> Loading the citation graph')
    citation_network, train_nodes, test_nodes = get_citation_graph(train_path, test_path)    
    node_label_map = get_node_label(file_path)

    # Calculate probabilities for guiding the random walks
    print('>> Computing random walk probabilities')
    prob_list = get_probs(citation_network, p=1, q=1)

    # Generate random walks
    print('>> Generating random walk')
    rand_walk = gen_rand_walk(citation_network, prob_list, 8, 10)

    # Generate embeddings using the random walks
    print('>> Generate Node2Vec embeddings')
    embeddings = generate_embeddings(rand_walk, 5, 100)

    # Prepare training and testing data
    print('>> Generating train and test splits')
    X_train, y_train, X_test, y_test = prepare_train_test_data(embeddings, train_nodes, test_nodes, node_label_map)

    # Train a logistic regression model
    print('>> Training Logistic Regression Model')
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    # Predict on the training and testing datasets
    train_preds = lr.predict(X_train)
    test_preds = lr.predict(X_test)

    # Evaluate the model performance
    print('>> Evaluating')
    accuracy_train = accuracy_score(y_train, train_preds)
    precision_train = precision_score(y_train, train_preds, average='macro')
    recall_train = recall_score(y_train, train_preds, average='macro')
    f1_train = f1_score(y_train, train_preds, average='macro')

    accuracy_test = accuracy_score(y_test, test_preds)
    precision_test = precision_score(y_test, test_preds, average='macro')
    recall_test = recall_score(y_test, test_preds, average='macro')
    f1_test = f1_score(y_test, test_preds, average='macro')

    # Print evaluation metrics for both training and testing datasets
    print(f'Train Metrics:\nAccuracy : {accuracy_train:.2f}, Precision : {precision_train:.2f}, Recall : {recall_train:.2f}, F1-Score : {f1_train:.2f}\n')
    print(f'Test Metrics:\nAccuracy : {accuracy_test:.2f}, Precision : {precision_test:.2f}, Recall : {recall_test:.2f}, F1-Score : {f1_test:.2f}\n')

    # Write the metrics to a file
    with open('lr_metrics.txt', 'w', encoding='utf-8') as file:
        file.write(f'Train Metrics:\nAccuracy : {accuracy_train:.2f}, Precision : {precision_train:.2f}, Recall : {recall_train:.2f}, F1-Score : {f1_train:.2f}\n')
        file.write(f'Test Metrics:\nAccuracy : {accuracy_test:.2f}, Precision : {precision_test:.2f}, Recall : {recall_test:.2f}, F1-Score : {f1_test:.2f}\n')
