import csv
from tqdm import tqdm
import pandas as pd
# louvain.
import networkx as nx
from community import community_louvain


def add_edge(graph, from_node, to_node):
    if from_node in graph:
        if to_node in graph[from_node]:
            graph[from_node][to_node] += 1
        else:
            graph[from_node][to_node] = 1
    else:
        graph[from_node] = {to_node: 1}

if __name__ == "__main__":
    path = 'emails_cleaned.csv'
    data = pd.read_csv(path)

    # only cotain receiver and sender
    nameList = ['X-From', 'X-To']
    senderReciverDF = data[nameList]

    # Graph construction via 2D array

    graph = {}

    # weighted graph
    for index, row in senderReciverDF.iterrows():
        add_edge(graph, row['X-From'], row['X-To'])

    # Test and print
    for sender, receivers in graph.items():
        for receiver, weight in receivers.items():
            print(f"{sender} -> {receiver} [Weight: {weight}]")

    # Cluster Consturction via louvain algorithm

    G = nx.Graph()
    for sender, receivers in graph.items():
        for receiver, weight in receivers.items():
            G.add_edge(sender, receiver, weight=weight)
    # louvain
    partition = community_louvain.best_partition(G)
    # louvain
    # Store node cluster information
    node_cluster = {node: cluster for node, cluster in partition.items()}

    # Transform the dictionary into the dataframe
    weightGraph_info = [(sender, receiver, weight) for sender, receivers in graph.items() for receiver, weight in
                        receivers.items()]

    weightGraph_df = pd.DataFrame(weightGraph_info, columns=['Sender', 'Receiver', 'Weight'])

    # if the reciver and sender name are belong to same cluster and add cluster id into this sepeate column in this row
    weightGraph_df['Cluster group'] = weightGraph_df.apply(
        lambda row: node_cluster[row['Sender']] if node_cluster[row['Sender']] == node_cluster[
            row['Receiver']] else None, axis=1)

    # Generate a new dataset
    weightGraphList = [weightGraph_df.columns.tolist()] + weightGraph_df.values.tolist()

    final_intermidate_save_name = 'Sender-receiver-weight-cluster.csv'

    with open(final_intermidate_save_name, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Wrap your loop with tqdm for the progress bar
        for row in tqdm(weightGraphList, desc="Writing rows"):
            writer.writerow(row)