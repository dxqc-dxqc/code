class DataGenerator:
    def generate_dataset(self):
        # Existing edges_dict construction
        edges_dict = {
            ('question', 'asked_by', 'user'),
            ('question', 'answered_by', 'user'),
            ('answer', 'in_question', 'question'),
            ('answer', 'rated_by', 'user'),
            ('user', 'similar_to', 'user'),
        }

        # Add reverse edges
        reverse_edges = {
            ('user', 'similar_to', 'user'),  # Bidirectional edge
            ('user', 'asked_by', 'question'),
            ('user', 'answered_by', 'question'),
            ('question', 'in_question', 'answer'),
            ('user', 'rated_by', 'answer'),
        }

        # Update edges_dict with reverse edges
        edges_dict.update(reverse_edges)
        
        return edges_dict
