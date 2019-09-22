import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree._tree import TREE_LEAF
from sklearn.tree import export_graphviz
import time	

class DTLearner:

    def __init__(self, 
        verbose = False,
        criterion = 'gini',
        splitter = 'best',
        max_depth = None,
        min_samples_split = 2,
        min_samples_leaf = 1,
        min_weight_fraction_leaf = 0.0,
        max_features = None,
        random_state = 1,
        max_leaf_nodes = None,
        min_impurity_decrease = 0.0,
        min_impurity_split = None,
        class_weight = None,
        presort = False):

        self.verbose = verbose
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.class_weight = class_weight
        self.presort = presort 

        self.classifier = DecisionTreeClassifier(
            criterion = self.criterion,
            splitter = self.splitter,
            max_depth = self.max_depth,
            min_samples_split = self.min_samples_split,
            min_samples_leaf = self.min_samples_leaf,
            min_weight_fraction_leaf = self.min_weight_fraction_leaf,
            max_features = self.max_features,
            random_state = self.random_state,
            max_leaf_nodes = self.max_leaf_nodes,
            min_impurity_decrease = self.min_impurity_decrease,
            min_impurity_split = self.min_impurity_split,
            class_weight = self.class_weight,
            presort = self.presort
        )

        pass


    def train(self, X_train, y_train):
        start = time.time()
        # Train Decision Tree Classifer
        self.classifier = self.classifier.fit(X_train, y_train)
        self.prune_index(self.classifier.tree_, 0, 5)
        end = time.time()
        diff = abs(end - start)
        print('train time taken for DT: ' + str(diff))

        """export_graphviz(
            self.classifier,
            out_file =  "tree.dot",
            feature_names = list(X_train.columns),
            filled = True,
            rounded = True)"""

    def query(self, X_test):
        start = time.time()
        #Predict the response for test dataset
        y_pred = self.classifier.predict(X_test)
        end = time.time()
        diff = abs(end - start)
        print('test time taken for DT: ' + str(diff))       
        
        return y_pred

    def prune_index(self, inner_tree, index, threshold):
        if inner_tree.value[index].min() < threshold:
            # turn node into a leaf by "unlinking" its children
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
        # if there are shildren, visit them as well
        if inner_tree.children_left[index] != TREE_LEAF:
            self.prune_index(inner_tree, inner_tree.children_left[index], threshold)
            self.prune_index(inner_tree, inner_tree.children_right[index], threshold)        


