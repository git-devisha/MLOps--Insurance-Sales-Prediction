import pandas as pd
import yaml
import os

class ingestion:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        # Load config relative to this file, not current working directory
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yml')
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
        
    def load_data(self):
        # Base directory = project root
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        train_data_path = os.path.join(base_dir, self.config['data']['train_path'])
        test_data_path = os.path.join(base_dir, self.config['data']['test_path'])

        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
        return train_data, test_data
