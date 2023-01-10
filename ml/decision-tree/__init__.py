import numpy as np 
import pandas as pd 

class Node():
    """This class will describe the nodes of our decision tree"""
    def __init__(self,feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        # ===== for decision node =====
        self.feature_index = feature_index
        self.threshold = threshold
        # access to left/right child nodes
        self.left = left
        self.right = right
        
        # store the information gain by the split denoted by the decision node
        self.info_gain = info_gain
        
        # ===== for leaf node =====
        self.value = value


class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        """ Decision Tree Classifier 
        args : 
        min_samples_split: minimum number of samples required to split an internal node
        max_depth: maximum depth of the tree
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
        # Initialize the root node of the tree
        self.root = None
        
        
    def build_tree(self, X, y, current_depth=0):
        """Builds the decision tree recursively by splitting the data at each node
        args:
        X: training data
        y: training labels
        current_depth: current depth of the tree
        """
        
        num_samples, num_features = np.shape(X)
        
        if num_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # find the best split
            best_split = self.get_best_split(X, y)
            
            # check if the best split is valid
            if best_split["info_gain"] > 0:
                # recursively build the left and right subtrees
                left_subtree = self.build_tree(best_split["left_X"], best_split["left_y"], current_depth + 1)
                right_subtree = self.build_tree(best_split["right_X"], best_split["right_y"], current_depth + 1)
                
                return Node(feature_index=best_split["feature_index"], threshold=best_split["threshold"], left=left_subtree, right=right_subtree, info_gain=best_split["info_gain"])
            
        # compute the leaf value 
        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value)
                
        
    def get_best_split(self, X, y):
        """Finds the best split for the data
        args:
        X: training data
        y: training labels
        """
        # dictionary to store the best split
        largest_info_gain = 0
        best_criteria = None
        
        # number of samples and features
        num_samples, num_features = np.shape(X)
        
        # find the best split
        for feature_index in range(num_features):
            feature_values = np.expand_dims(X[:, feature_index], axis=1)
            unique_values = np.unique(feature_values)
            
            # loop through all the unique values of the feature and split the data
            for threshold in unique_values:
                # split the data
                X_left, y_left, X_right, y_right = self.split(X, y, feature_index, threshold)
                
                # skip if the split does not divide the dataset
                if len(X_left) == 0 or len(X_right) == 0:
                    continue
                
                # calculate the information gain
                current_info_gain = self.information_gain(y, y_left, y_right, "gini")
                
                # update the best criteria if the information gain is higher
                if current_info_gain > largest_info_gain:
                    largest_info_gain = current_info_gain
                    best_criteria = {"feature_index": feature_index, "threshold": threshold}
                    best_sets = {"left_X": X_left, "left_y": y_left, "right_X": X_right, "right_y": y_right}
                    
        return best_criteria, best_sets
    
    def split(self, X, y, feature_index, threshold):
        """Splits the data into two sets based on the feature index and threshold
        args:
        X: training data
        y: training labels
        feature_index: index of the feature to split on
        threshold: value of the feature to split on
        """
        # split the data
        X_left = X[X[:, feature_index] <= threshold]
        y_left = y[X[:, feature_index] <= threshold]
        X_right = X[X[:, feature_index] > threshold]
        y_right = y[X[:, feature_index] > threshold]
        
        return X_left, y_left, X_right, y_right
    
    def information_gain(self, y, y_left, y_right, criterion):
        """Calculates the information gain
        args:
        y: training labels
        y_left: labels of the left split
        y_right: labels of the right split
        criterion: the criterion to use to calculate the information gain
        """
        # calculate the information gain
        if criterion == "gini":
            info_gain = self.gini(y) - (len(y_left)/len(y))*self.gini(y_left) - (len(y_right)/len(y))*self.gini(y_right)
        elif criterion == "entropy":
            info_gain = self.entropy(y) - (len(y_left)/len(y))*self.entropy(y_left) - (len(y_right)/len(y))*self.entropy(y_right)
        else:
            raise ValueError("Invalid criterion")
            
        return info_gain
    
    def entropy(self, y):
        """Calculates the entropy
        args:
        y: training labels
        """
        # calculate the entropy
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = sum(probabilities * -np.log2(probabilities))
        
        return entropy
    
    def gini(self, y):
        """Calculates the gini impurity
        args:
        y: training labels
        """
        # calculate the gini impurity
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        gini = 1 - sum(probabilities**2)
        
        return gini
    
    def calculate_leaf_value(self, y):
        """Calculates the value of a leaf node
        args:
        y: training labels
        """
        # calculate the leaf value
        leaf_value = np.mean(y)
        
        return leaf_value
    
    def print_tree(self, node, spacing=" "):
        """Prints the decision tree
        args:
        node: current node
        spacing: spacing to add before printing the node
        """
        # base case
        if node.value is not None:
            print(spacing + "Predict", node.value)
            return
        
        # print the current node
        print(spacing + "X{} <= {}".format(node.feature_index, node.threshold))
        
        # print the left subtree
        print(spacing + "--> True:")
        self.print_tree(node.left, spacing + "  ")
        
        # print the right subtree
        print(spacing + "--> False:")
        self.print_tree(node.right, spacing + "  ")
        
    def fit(self, X, y):
        """Builds the decision tree
        args:
        X: training data
        y: training labels
        """
        # build the tree
        self.root = self.build_tree(X, y)
        
    def predict(self, x, node):
        """Predicts the value of a single sample
        args:
        x: sample
        node: current node
        """
        # base case
        if node.value is not None:
            return node.value
        
        # determine the next node
        if x[node.feature_index] <= node.threshold:
            return self.predict_value(x, node.left)
        else:
            return self.predict_value(x, node.right)
    
    def make_predictions(self, X, node):
        """Makes predictions for a set of samples
        args:
        X: samples
        node: current node
        """
        # make predictions
        y_pred = [self.predict(x, node) for x in X]
        
        return y_pred

data = pd.read_csv("ml/decision-tree/iris.csv", skiprows=1, names=["sepal_length", "sepal_width", "petal_length", 'petal_width', 'type'])
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
classifier.fit(X_train, y_train)
classifier.print_tree(classifier.root)

y_pred = classifier.predict(X_test, classifier.root)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)