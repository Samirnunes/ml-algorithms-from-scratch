import pandas as pd
from node import Node
from id3 import ID3

class DecisionTree:
    def __init__(self):
        self.__root_node = None 
        self.__algorithm = ID3()
        self.__trained = False
        
    def __str__(self):
        if self.__trained:
            level = 0
            print("E=Entropy, I=Information Gain, SF=Split Feature, SV=Split Value Threshold (True node if >=), n=Number of Samples \n")
            entropy = round(self.__root_node.entropy, 2) if self.__root_node.entropy is not None else None
            splits_info_gain = round(self.__root_node.splits_info_gain, 2) if self.__root_node.splits_info_gain is not None else None
            split_feature = self.__root_node.split_feature
            split_feature_value = round(self.__root_node.split_feature_value, 2) if self.__root_node.split_feature_value is not None else None
            string = "\t"*level + "Root" + ": " + f"E={entropy}" + ", " + \
                    f"I={splits_info_gain}" + ", " + \
                    f"SF={split_feature}" + ", " + \
                    f"SV={split_feature_value}" + "\n"
            if self.__root_node.true_child is not None and self.__root_node.false_child is not None:
                level += 1
                string += self.__str_recursive(level, self.__root_node.true_child, "True")
                string += self.__str_recursive(level, self.__root_node.false_child, "False")
            return string
        return None
        
    def __str_recursive(self, level, node: Node, node_type: str):
        entropy = round(node.entropy, 2) if node.entropy is not None else None
        splits_info_gain = round(node.splits_info_gain, 2) if node.splits_info_gain is not None else None
        split_feature = node.split_feature
        split_feature_value = round(node.split_feature_value, 2) if node.split_feature_value is not None else None
        n_samples = node.num_samples
        string = "\t"*level + node_type + ": " + f"E={entropy}" + ", " + \
                f"I={splits_info_gain}" + ", " + \
                f"SF={split_feature}" + ", " + \
                f"SV={split_feature_value}" + ", " + \
                f"n={n_samples}" + "\n"
        if node.true_child is not None and node.false_child is not None:
            level += 1
            string += self.__str_recursive(level, node.true_child, "True")
            string += self.__str_recursive(level, node.false_child, "False")
        return string
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        if type(y) == pd.DataFrame:
            target = y.columns[0]
        else:
            target = y.name
        dataset = pd.concat([X, y], axis=1)
        self.__root_node = self.__algorithm.build_tree(dataset, target)
        self.__trained = True
    
    def predict(self, X: pd.DataFrame):
        node = self.__root_node
        predictions = []
        if self.__trained:
            for row in X.index:
                if node.true_child is None or node.false_child is None:
                    predictions.append(node.prediction)
                else:
                    decision_feature = node.split_feature
                    decision_value = node.split_feature_value
                    next_node = node.true_child if X.loc[row][decision_feature] >= decision_value else node.false_child
                    predictions.append(self.__predict_recursive(X, row, next_node))
            return predictions
        return None
    
    def __predict_recursive(self, X: pd.DataFrame, row, node: Node):
        if node.true_child is None or node.false_child is None:
            return node.prediction
        decision_feature = node.split_feature
        decision_value = node.split_feature_value
        next_node = node.true_child if X.loc[row][decision_feature] >= decision_value else node.false_child
        return self.__predict_recursive(X, row, next_node)