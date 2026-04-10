import os
import random
import numpy as np
import torch 
import torch.nn as nn 
import yaml


def load_config(path='config.yaml'):
    config_dir = os.path.dirname(os.path.realpath(__file__)) 
    config_path = os.path.join(config_dir, path) 
    
    if not os.path.exists(config_path): 
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_subconfig(section, path='config.yaml'):
    config = load_config(path)
    return config.get(section, {})


def xavier_init(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight)
        

def kaiming_init(model):
    if isinstance(model, nn.Linear):
        nn.init.kaiming_uniform_(model.weight)  


def build_class_to_topclass_mapping(class_dict, top_class_dict):
    class_to_topclass = {}

    for class_name, class_id in class_dict.items():
        for top_class_name, top_class_id in top_class_dict.items():
            if class_name.startswith(top_class_name):
                class_to_topclass[class_id] = top_class_id
                break

    return class_to_topclass


def build_class_to_topclass_tensor(class_dict, top_class_dict, device):
    num_classes = len(class_dict)
    class_to_topclass = torch.zeros(num_classes, dtype=torch.long, device=device)
    
    for class_name, class_id in class_dict.items():
        for top_class_name, top_class_id in top_class_dict.items():
            if class_name.startswith(top_class_name):
                class_to_topclass[class_id] = top_class_id
                break
    
    return class_to_topclass


def build_id_to_class_mapping(class_dict):
    """Return a mapping from class ID to class name."""
    return {class_id: class_name for class_name, class_id in class_dict.items()}


def extend_subcat(subcat):
    # Returns a tuple of (top-level, subcat) for a given subcat.
    if "-" not in subcat:
        raise Exception("invalid subcat name: " + subcat + ", top level not found <top>-<subcat>")
    top_level = subcat.split("-")[0]
    return (top_level, subcat)


def get_top_level(subcat):
  # Returns top-level of a given subcat
  return extend_subcat(subcat)[0]


def intersection(list1, list2):
  # Returns a list containing common elements in two other lists
  return list(set(list1).intersection(list2))


def set_seed(seed=1821):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed

        
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
