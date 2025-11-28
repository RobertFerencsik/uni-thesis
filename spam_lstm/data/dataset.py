# spam_lstm/data/dataset.py

import pandas as pd

class SpamDataset:
    def __init__(self, csv_path: str):
        self.data = pd.read_csv(csv_path)

    def get_dataframe(self):
        return self.data.copy()
