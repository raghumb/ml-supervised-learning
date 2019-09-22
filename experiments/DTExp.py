from experiments.ExperimentHelper import ExperimentHelper
from learners.DTLearner import DTLearner
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, make_scorer

class DTExp:
    def __init__(self, reader, helper, splitter):
        self.reader = reader
        self.helper = helper
        self.learner = DTLearner()
        self.splitter = splitter
        self.expHelper = ExperimentHelper(self.splitter, self.learner)

    def experiment(self):

        self.model_complexity_exp()
        if(self.splitter.reader.dataset == 'Bank'):
            print('bank')  
            self.learning_curve_iter2_bank()
            self.experiment_run_test_bank()
        else:
            self.learning_curve_iter2_wine()            
            self.experiment_run_test_wine()
        
        # Perform learning curve
        #self.expHelper.learning_curve_exp()
        #self.model_complexity_exp()
        #self.exp_grid_search()
        #self.learning_curve_iter2()
        #self.learning_curve_iter2()
        #self.learner.train(self.splitter.X_train, self.splitter.y_train)
        #y_pred = self.learner.query(self.splitter.X_test)
        """print("Final Accuracy for " + str(self.learner.__class__)+": ", 
                        metrics.accuracy_score(self.splitter.y_test, y_pred))"""

    def model_complexity_exp(self):
        #TODO should we create a new learner object??
        self.learner = DTLearner()
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        param_range = np.array([2,4,6,8])
        self.expHelper.model_complexity_exp('max_depth', param_range)

        self.learner = DTLearner(max_depth = 4)
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        param_range = np.array([2,5,6,8,10])
        self.expHelper.model_complexity_exp('max_leaf_nodes', param_range)

        """self.learner = DTLearner(max_depth = 4, max_leaf_nodes = 5)
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        param_range = np.array([1,2,5,6,8,10])
        self.expHelper.model_complexity_exp('min_samples_leaf', param_range)    """       

    def exp_grid_search(self):
        max_leaf_nodes_range = list(range(2, 20))
        max_depth_range = list(range(2, 20))
        min_samples_split_range = list(range(2, 5))
        min_samples_leaf_range = list(range(2, 5))
        grid_param = dict(max_leaf_nodes = max_leaf_nodes_range, 
                         max_depth = max_depth_range, 
                         min_samples_split = min_samples_split_range,
                         min_samples_leaf = min_samples_leaf_range)

        self.expHelper.perform_grid_search(grid_param)    

    def learning_curve_iter2_bank(self):
        self.learner = DTLearner(
                    max_depth = 4,
                    max_leaf_nodes = 5
        )
        self.expHelper = ExperimentHelper(self.splitter, self.learner, '-iter-2')
        self.expHelper.learning_curve_exp()

    def learning_curve_iter2_wine(self):
        self.learner = DTLearner(
                    max_depth = 4,
                    max_leaf_nodes = 5
        )
        self.expHelper = ExperimentHelper(self.splitter, self.learner, '-iter-2')
        self.expHelper.learning_curve_exp()        

    def experiment_run_test_wine(self): 
        self.learner = DTLearner(
                    max_depth = 4,
                    max_leaf_nodes = 5
        )
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()         

    def experiment_run_test_bank(self): 
        self.learner = DTLearner(
                    max_depth = 4,
                    max_leaf_nodes = 5
        )
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()         