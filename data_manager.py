import pandas as pd
import numpy as np
import os

class DataLoader:
    """Handles reading CSV files for the ML pipeline."""
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def load_data(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        return pd.read_csv(filepath)
