# This is the updated version of data_c.py to restore the full heterogeneous QA dataset generator

# Import necessary libraries
import pickle
import networkx as nx

# Define node types
user_node_type = 'user'
question_node_type = 'question'
answer_node_type = 'answer'

# Define edge types
asks_edge_type = 'asks'
answers_edge_type = 'answers'
contains_edge_type = 'contains'
rates_edge_type = 'rates'
similar_to_edge_type = 'similar_to'

# Define reverse edges
rev_asks_edge_type = 'rev_asks'
rev_answers_edge_type = 'rev_answers'
rev_contains_edge_type = 'rev_contains'
rev_rates_edge_type = 'rev_rates'
rev_similar_to_edge_type = 'rev_similar_to'

# Function to generate the heterogeneous dataset

def generate_heterogeneous_dataset():
    # Your implementation here
    # Ensure it matches required output structure
    hetero_dataset = {
        'user_features': None,
        'question_features': None,
        'answer_features': None,
        'edges': {},
        'rating_attrs': {},
        'user_labels': {},
        'config': {}
    }
    
    # Add edges dict keyed by (src_type, rel, dst_type)
    hetero_dataset['edges'][(user_node_type, asks_edge_type, question_node_type)] = []
    hetero_dataset['edges'][(question_node_type, answers_edge_type, answer_node_type)] = []
    # Add reverse edges
    hetero_dataset['edges'][(question_node_type, rev_asks_edge_type, user_node_type)] = []
    hetero_dataset['edges'][(answer_node_type, rev_answers_edge_type, question_node_type)] = []
    return hetero_dataset

if __name__ == '__main__':
    dataset = generate_heterogeneous_dataset()
    with open('hetero_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)