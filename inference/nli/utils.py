import json
import pandas as pd

def read_json(file_path):
    # with open(file_path) as f:
    #     return json.load(f)
    return pd.read_csv(file_path)
