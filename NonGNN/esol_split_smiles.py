import json
import pandas as pd

def load_split(json_path):
    with open(json_path, 'r') as f:
        split = json.load(f)
    return split

fold_indices = load_split('clintox_split_5fold.json')

dataset = pd.read_csv("clintox.csv")
smiles_list = dataset['smiles']

fold_values = []

for fold in fold_indices:
    values = {}
    values["train_smiles"] = smiles_list[fold["train_idx"]].to_list()
    values["val_smiles"] = smiles_list[fold["val_idx"]].to_list()
    values["test_smiles"] = smiles_list[fold["test_idx"]].to_list()
    fold_values.append(values)



with open('clintox_split_smiles.json', 'w') as f:
    json.dump(fold_values, f)