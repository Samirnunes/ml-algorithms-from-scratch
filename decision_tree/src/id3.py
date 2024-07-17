import pandas as pd
import numpy as np
from node import Node

class ID3:
    def __init__(self):
        pass
    
    def build_tree(self, dataset: pd.DataFrame, target: str):
        root_node = Node(dataset, target, None)
        if len(dataset[target].unique()) > 1 and len(dataset.columns) > 1:
            true_split, false_split = self.__split(root_node)
            if true_split is not None and false_split is not None:
                root_node.true_child = self.__build_tree_recursive(true_split, target, root_node)
                root_node.false_child = self.__build_tree_recursive(false_split, target, root_node)
        return root_node
    
    def __build_tree_recursive(self, split: pd.DataFrame, target: str, parent_node: Node):
        node = Node(split, target, parent_node)
        if len(split[target].unique()) > 1 and len(split.columns) > 1:
            true_split, false_split = self.__split(node)
            if true_split is not None and false_split is not None:
                node.true_child = self.__build_tree_recursive(true_split, target, node)
                node.false_child = self.__build_tree_recursive(false_split, target, node)
        return node
    
    def __split(self, node: Node):
        dataset = node.split
        target = node.target
        features = dataset.drop([target], axis=1).columns
        best_info_gain = 0
        best_feature = None
        best_true_split = None
        best_false_split = None
        for feature in features:
            ordered_feature_values = sorted(list(dataset[feature].unique()))
            if len(ordered_feature_values) > 1:
                for feature_value in ordered_feature_values:
                    info_gain = node.entropy
                    true_split = dataset[dataset[feature] >= feature_value].drop([feature], axis=1)
                    false_split = dataset[dataset[feature] < feature_value].drop([feature], axis=1)
                    if len(true_split) > 0 and len(false_split) > 0:       
                        true_node = Node(true_split, target, node)
                        false_node = Node(false_split, target, node)
                        info_gain -= (true_node.num_samples/node.num_samples) * true_node.entropy
                        info_gain -= (false_node.num_samples/node.num_samples) * false_node.entropy
                        if info_gain > best_info_gain:
                            best_info_gain = info_gain
                            best_feature = feature
                            best_feature_value = feature_value
                            best_true_split = true_split
                            best_false_split = false_split
            else:
                node.split.drop([feature], axis=1, inplace=True)
        if best_true_split is not None and best_false_split is not None:
            node.splits_info_gain = best_info_gain
            node.split_feature = best_feature
            node.split_feature_value = best_feature_value
        return best_true_split, best_false_split