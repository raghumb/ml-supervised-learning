from experiments.ExperimentHelper import ExperimentHelper
from learners.BoostLearner import BoostLearner
from learners.DTLearner import DTLearner
import numpy as np

class BoostingExp:
    def __init__(self, reader, helper, splitter):
        self.reader = reader
        self.helper = helper
        self.learner = BoostLearner()
        self.splitter = splitter
        self.expHelper = ExperimentHelper(self.splitter, self.learner)

    def experiment(self):

        # Perform learning curve
        #self.expHelper.learning_curve_exp()
        self.model_complexity_exp()
        self.model_complexity_exp2()
        self.learning_curve_iter2()
        if(self.splitter.reader.dataset == 'Bank'):
            print('bank')            
            self.learning_curve_iter2_bank()
            self.experiment_run_test_bank()
        else:
            self.experiment_run_test_wine()
            self.learning_curve_iter2_wine()
            


        #self.learner.train(self.splitter.X_train, self.splitter.y_train)
        #y_pred = self.learner.query(self.splitter.X_test)
        """print("Final Accuracy for " + str(self.learner.__class__)+": ", 
                        metrics.accuracy_score(self.splitter.y_test, y_pred))"""

    def model_complexity_exp(self):
        #TODO should we create a new learner object??
        
        self.learner = BoostLearner()
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        param_range = np.array([2,4,6,8, 9,10])
        self.expHelper.model_complexity_exp('max_depth', param_range)

    def model_complexity_exp_2(self):
        #TODO should we create a new learner object??
        
        self.learner = BoostLearner(max_depth = 2)
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        param_range = np.array([0.1, 0.2,0.4,0.6,0.8, 0.9])
        self.expHelper.model_complexity_exp('learning_rate', param_range)
         



    def learning_curve_iter2_wine(self):
        self.learner = BoostLearner(max_depth = 9, learning_rate =0.4)
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper = ExperimentHelper(self.splitter, self.learner, '-iter-2')
        self.expHelper.learning_curve_exp() 

    def learning_curve_iter2_bank(self):
        self.learner = BoostLearner(max_depth = 2, learning_rate =0.1)
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper = ExperimentHelper(self.splitter, self.learner, '-iter-2')
        self.expHelper.learning_curve_exp()         

    def experiment_run_test_wine(self): 
        self.learner = BoostLearner(max_depth = 9, learning_rate =0.4)
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()                

    def experiment_run_test_bank(self): 
        self.learner = BoostLearner(max_depth = 2, learning_rate =0.1)
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()  