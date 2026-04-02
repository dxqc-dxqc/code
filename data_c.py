# Imports
import pickle
import random

# Configuration
class DatasetConfig:
    def __init__(self):
        self.users = 1000
        self.items = 1000
        self.similarity_types = ['similar_to']

# Data Generator
class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.data = self.generate_data()

    def generate_data(self):
        # Generate user and item data
        users = {f'user_{i}': [] for i in range(self.config.users)}
        items = {f'item_{i}': [] for i in range(self.config.items)}
        edges = []

        # Generate edges
        for user in users:
            for i in range(random.randint(0, 10)):
                item = f'item_{random.randint(0, self.config.items - 1)}'
                edges.append((user, 'similar_to', item))
                edges.append((item, 'similar_to', user))  # Reverse edge

        return edges

# Main guard
if __name__ == '__main__':
    config = DatasetConfig()
    generator = DataGenerator(config)
    with open('hetero_dataset.pkl', 'wb') as f:
        pickle.dump(generator.data, f)