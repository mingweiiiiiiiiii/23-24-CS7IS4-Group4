import time
import json
import threading
import concurrent.futures

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def process_similarity(source_node, fp, file_lock):
    start_time = time.perf_counter()
    source_data = node_list[source_node]
    source_paper_id = source_data['id']
    similarities = cosine_similarity(abstract_matrix[source_node], abstract_matrix)
    sim_end_time = time.perf_counter()
    print(f"Thread {threading.get_ident()}, source node {source_node}: Calculated similarities in {sim_end_time - start_time} seconds")
    with file_lock:
        for target_node, cos_sim in enumerate(similarities):
            target_paper_id = json.loads(papers[target_node])['paper_id']
            if target_paper_id != source_paper_id:
                fp.write(f"{source_paper_id},{target_paper_id},{cos_sim}\n")
    end_time = time.perf_counter()
    print(f"Thread {threading.get_ident()}, source node {source_node}: Total execution time is {end_time - start_time} seconds")

# Load the citation data
with open('./data/citation_relations.json', 'r') as f:
    cite_data = json.load(f)

# Create a list of nodes
node_list = []
for paper_id in cite_data.keys():
    node_list.append({ 'id': paper_id })

# Load the paper data
with open('./data/papers.SSN.jsonl', 'r') as f:
    papers = f.readlines()

# Implement TF-IDF on the paper abstracts
abstracts = []
for paper in papers:
    paper_data = json.loads(paper)
    abstracts.append(" ".join(paper_data['abstract']))

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
abstract_matrix = vectorizer.fit_transform(abstracts)

# Create a file lock to prevent race conditions
file_lock = threading.Lock()

# Create a ThreadPoolExecutor to calculate similarities in parallel
num_threads = 20
with open('./similarities.csv', 'w') as fp:
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Create a new thread for each source node
        threads = [executor.submit(process_similarity, source_node, fp, file_lock) for source_node in range(len(node_list))]
        concurrent.futures.wait(threads)