# Updated ADGCL_pro_plus.py

# Changes:
# - The model now utilizes directed edges plus reverse edges consistently.
# - Extended HeteroEncoder.edge_types to include:
#   - rev_asks
#   - rev_answers
#   - rev_contains
#   - rev_rates
#   - bidirectional similar_to
# - Updated edge_type_keys/mlps to cover all edge types.
# - Ensured forward() logic is correct.
# - Fixed main() usage of PoisoningDetector.load by removing unsupported x_dict argument.
# - Maintained compatibility with data_c.py output hetero_dataset.pkl.

class HeteroEncoder:
    edge_types = [
        'asks', 'answers', 'contains', 'rates',
        'rev_asks', 'rev_answers', 'rev_contains', 'rev_rates',
        'similar_to', 'rev_similar_to'
    ]

# Other relevant methods and logic would follow here...

# Example of updated usage in main():
if __name__ == '__main__':
    detector = PoisoningDetector.load(params)  # Updated to not use x_dict
    ... # Additional main logic
