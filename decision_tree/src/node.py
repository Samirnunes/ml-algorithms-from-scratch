import pandas as pd
import numpy as np

class Node:
    def __init__(self, split: pd.DataFrame, target: str = None, parent_node = None):
        self.split = split
        self.num_samples = len(split)
        self.parent_node = parent_node
        self.true_child = None
        self.false_child = None
        self.target = target
        self.prediction = split[target].value_counts().idxmax()
        self.entropy = self.__target_entropy()
        self.split_feature = None
        self.split_feature_value = None
        self.splits_info_gain = None
        
    def __target_entropy(self):
        entropy = 0
        target_col = self.split[self.target]
        class_counts = target_col.value_counts()
        for count in class_counts:
            p = count/self.num_samples
            entropy -= p * np.log2(p)
        return entropy