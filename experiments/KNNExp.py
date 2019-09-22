from experiments.ExperimentHelper import ExperimentHelper
from learners.KNNLearner import KNNLearner
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, make_scorer

class KNNExp:
    def __init__(self, reader, helper, splitter):
        self.reader = reader
        self.helper = helper
        self.learner = KNNLearner()
        self.splitter = splitter
        self.expHelper = ExperimentHelper(self.splitter, self.learner)

    def experiment(self):
        self.model_complexity_exp1()
        self.model_complexity_exp11()
        self.model_complexity_exp2()
        self.model_complexity_exp3()
        if(self.splitter.reader.dataset == 'Bank'):
            print('bank')  
            self.experiment_run_test_bank()        
            self.learning_curve_iter2_bank()
        else:
            self.experiment_run_test_wine()
            self.learning_curve_iter2_wine()
        
        
        
        # Perform learning curve
        #self.expHelper.learning_curve_exp()        
        #self.model_complexity_exp1()
        #self.model_complexity_exp2()
        #self.model_complexity_exp3()
        #self.model_complexity_exp4()
        """"self.exp_grid_search()"""
        #self.learning_curve_iter2()

        #self.learner.train(self.splitter.X_train, self.splitter.y_train)
        #y_pred = self.learner.query(self.splitter.X_test)
        """print("Final Accuracy for " + str(self.learner.__class__)+": ", 
                        metrics.accuracy_score(self.splitter.y_test, y_pred))"""

    def model_complexity_exp1(self):
        #TODO should we create a new learner object??
        self.learner = KNNLearner()
        self.expHelper = ExperimentHelper(self.splitter, self.learner, 'euclidean')
        #param_range = np.array([1,2,3,4,5])
        param_range = np.arange(1, 40, 2)
        print(param_range)
        self.expHelper.model_complexity_exp('n_neighbors', param_range)

    def model_complexity_exp11(self):
        #TODO should we create a new learner object??
        self.learner = KNNLearner()
        self.expHelper = ExperimentHelper(self.splitter, self.learner, 'euclidean', '1')
        #param_range = np.array([1,2,3,4,5])
        param_range = np.arange(40, 60, 2)
        print(param_range)
        self.expHelper.model_complexity_exp('n_neighbors', param_range)        


    def model_complexity_exp2(self):
        #TODO should we create a new learner object??
        self.learner = KNNLearner(metric = 'euclidean')
        self.expHelper = ExperimentHelper(self.splitter, self.learner, 'euclidean')
        param_range = np.array([1,2,3,4,5])
        self.expHelper.model_complexity_exp('n_neighbors', param_range)

    def model_complexity_exp3(self):
        #TODO should we create a new learner object??
        self.learner = KNNLearner(weights = 'distance')
        self.expHelper = ExperimentHelper(self.splitter, self.learner, 'distance weight')
        param_range = np.array([1,2,3,4,5])
        self.expHelper.model_complexity_exp('n_neighbors', param_range)

    def model_complexity_exp4(self):
        #TODO should we create a new learner object??
        self.learner = KNNLearner(metric = 'euclidean', weights = 'distance')
        self.expHelper = ExperimentHelper(self.splitter, self.learner, 'distance weight')
        param_range = np.arange(1, 40, 2)
        print(param_range)
        self.expHelper.model_complexity_exp('n_neighbors', param_range)

    def exp_grid_search(self):
        k_range = list(range(1, 31))
        weight_options = ['uniform', 'distance']
        metric_options = ['euclidean', 'minkowski']
        grid_param = dict(n_neighbors = k_range, weights = weight_options, metric = metric_options)

        self.expHelper.perform_grid_search(grid_param)        



        """grid_param = {
            'n_estimators': [100, 300, 500, 800, 1000],
            'criterion': ['gini', 'entropy'],
            'bootstrap': [True, False]
        }"""        


    def learning_curve_iter2_wine(self):
        self.learner = KNNLearner(
                        metric = 'euclidean',
                        n_neighbors = 30,
                        weights = 'uniform'
                        )
        self.expHelper = ExperimentHelper(self.splitter, self.learner, '-iter-2')
        self.expHelper.learning_curve_exp()

    def learning_curve_iter2_bank(self):
        self.learner = KNNLearner(
                        metric = 'euclidean',
                        n_neighbors = 38,
                        weights = 'uniform'
                        )
        self.expHelper = ExperimentHelper(self.splitter, self.learner, '-iter-2')
        self.expHelper.learning_curve_exp()        

    def experiment_run_test_wine(self): 
        self.learner = KNNLearner(
                        metric = 'euclidean',
                        n_neighbors = 30,
                        weights = 'uniform'
                        ) 
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()                                

    def experiment_run_test_bank(self): 
        self.learner = KNNLearner(
                        metric = 'euclidean',
                        n_neighbors = 38,
                        weights = 'uniform'
                        ) 
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()        