from experiments.ExperimentHelper import ExperimentHelper
from learners.SVMLearner import SVMLearner
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, make_scorer

class SVMExp:
    def __init__(self, reader, helper, splitter):
        self.reader = reader
        self.helper = helper
        self.learner = SVMLearner()
        self.splitter = splitter
        self.expHelper = ExperimentHelper(self.splitter, self.learner)

    def experiment(self):

        #self.model_complexity_exp5()
        #self.model_complexity_exp6()
        #self.model_complexity_exp4()
        #self.model_complexity_exp44()
        #self.model_complexity_exp2()
        self.model_complexity_exp6()

        # Perform learning curve
        #self.expHelper.learning_curve_exp()
        """self.model_complexity_exp1()
        self.model_complexity_exp2()
        self.model_complexity_exp3()
        self.model_complexity_exp4()"""
        #self.model_complexity_exp5()
        #self.model_complexity_exp6()
        """self.exp_grid_search()"""
        #self.learning_curve_iter2()
        #self.model_complexity_exp4()
        #self.model_complexity_exp44()

        #self.learner.train(self.splitter.X_train, self.splitter.y_train)
        #y_pred = self.learner.query(self.splitter.X_test)
        """print("Final Accuracy for " + str(self.learner.__class__)+": ", 
                        metrics.accuracy_score(self.splitter.y_test, y_pred))"""

    
        
    def model_complexity_exp1(self):
        #TODO should we create a new learner object??
        self.learner = SVMLearner()
        self.expHelper = ExperimentHelper(self.splitter, self.learner, 'rbf')
        param_range = np.array([2,4,6,8])
        self.expHelper.model_complexity_exp('degree', param_range)


    def model_complexity_exp2(self):
        #TODO should we create a new learner object??
        self.learner = SVMLearner(kernel = 'linear', C = 0.2)
        self.expHelper = ExperimentHelper(self.splitter, self.learner, 'linear4', 'Linear Kernel')
        param_range = np.array([1,2,4,5, 6,7,8])
        self.expHelper.model_complexity_exp('gamma', param_range)

        self.learner = SVMLearner(kernel = 'linear', C = 0.2)
        self.expHelper = ExperimentHelper(self.splitter, self.learner, 'linear5', 'Linear Kernel')
        param_range = np.array([0.1,0.2,0.4,0.5, 0.6,0.7,0.8])
        self.expHelper.model_complexity_exp('gamma', param_range)        

    def model_complexity_exp3(self):
        #TODO should we create a new learner object??
        self.learner = SVMLearner(kernel = 'sigmoid')
        self.expHelper = ExperimentHelper(self.splitter, self.learner, 'sigmoid', 'Sigmoid Kernel')
        #param_range = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
        #self.expHelper.model_complexity_exp('C', param_range)        

        param_range = np.array([0.01, 0.03, 0.05, 0.07, 0.09])
        self.expHelper.model_complexity_exp('gamma', param_range) 


    def model_complexity_exp4(self):
        #TODO should we create a new learner object??
        self.learner = SVMLearner(kernel = 'linear')
        self.expHelper = ExperimentHelper(self.splitter, self.learner, 'linear3','Linear Kernel')
        #param_range = np.array([0.5, 1, 1.5, 2])
        #param_range = np.array([2, 3, 4, 5, 6, 9, 12])
        #param_range = np.array([0.01, 0.02, 0.03, 0.043, 0.05, 0.06])
        param_range = np.array([0.0001, 0.0005, 0.0008, 0.001, 0.005, 0.008])
        self.expHelper.model_complexity_exp('C', param_range) 

    def model_complexity_exp44(self):
        #TODO should we create a new learner object??
        self.learner = SVMLearner(kernel = 'linear')
        self.expHelper = ExperimentHelper(self.splitter, self.learner, 'linear1','Linear Kernel')
        #param_range = np.array([0.5, 1, 1.5, 2])
        #param_range = np.array([2, 3, 4, 5, 6, 9, 12])
        param_range = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1])
        self.expHelper.model_complexity_exp('C', param_range) 

        """self.learner = SVMLearner(kernel = 'linear')
        self.expHelper = ExperimentHelper(self.splitter, self.learner, 'linear3', 'Linear Kernel')
        
        param_range = np.array([2, 3, 4, 5, 6, 9, 12])   
        self.expHelper.model_complexity_exp('C', param_range) """               

    def model_complexity_exp5(self):
        #TODO should we create a new learner object??
        self.learner = SVMLearner(kernel = 'rbf')
        self.expHelper = ExperimentHelper(self.splitter, self.learner, 'rbf1', 'rbf Kernel')

        param_range = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
        self.expHelper.model_complexity_exp('C', param_range) 

        self.learner = SVMLearner(kernel = 'rbf')
        self.expHelper = ExperimentHelper(self.splitter, self.learner, 'rbf2', 'rbf Kernel')
        param_range = np.array([0.5, 1, 1.5, 2])

        self.expHelper.model_complexity_exp('C', param_range) 

        self.learner = SVMLearner(kernel = 'rbf')
        self.expHelper = ExperimentHelper(self.splitter, self.learner, 'rbf3', 'rbf Kernel')
        
        param_range = np.array([2, 3, 4, 5, 6, 9, 12])
        
        self.expHelper.model_complexity_exp('C', param_range)                 

    def model_complexity_exp6(self):
        #TODO should we create a new learner object??
        #self.learner = SVMLearner(kernel = 'rbf', C =0.1)
        self.learner = SVMLearner(kernel = 'rbf', C =0.5)
        self.expHelper = ExperimentHelper(self.splitter, self.learner, 'rbf8','rbf Kernel')
        #param_range = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        #param_range = np.array([0.5,2, 3, 4, 5, 6])
        param_range = np.array([0.01, 0.03, 0.05, 0.07, 0.09])
        self.expHelper.model_complexity_exp('gamma', param_range) 

        self.learner = SVMLearner(kernel = 'rbf', C =0.5)
        self.expHelper = ExperimentHelper(self.splitter, self.learner, 'rbf9','rbf Kernel')
        param_range = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        #param_range = np.array([0.5,2, 3, 4, 5, 6])
        #param_range = np.array([0.01, 0.03, 0.05, 0.07, 0.09])
        self.expHelper.model_complexity_exp('gamma', param_range)  

        """self.learner = SVMLearner(kernel = 'rbf', C =0.2)
        self.expHelper = ExperimentHelper(self.splitter, self.learner, 'rbf5','rbf Kernel')
        #param_range = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        param_range = np.array([0.5,2, 3, 4, 5, 6])
        #param_range = np.array([0.01, 0.03, 0.05, 0.07, 0.09])
        self.expHelper.model_complexity_exp('gamma', param_range)"""                              

    def exp_grid_search(self):
        #degree_range = list(range(1, 31))
        kernel_options = ['rbf']
        c_range = np.array([1, 2, 3, 4, 5, 6, 9, 12])  
        gamma_options = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
        grid_param = dict(C = c_range, kernel = kernel_options, gamma = gamma_options)

        self.expHelper.perform_grid_search(grid_param)         

    def learning_curve_iter2_wine(self):
        self.learner = SVMLearner(
                        gamma = 0.05,
                        C=0.5,
                        kernel = 'rbf'
                        )
        self.expHelper = ExperimentHelper(self.splitter, self.learner, '-iter-2')
        self.expHelper.learning_curve_exp() 

    def learning_curve_iter2_bank(self):
        self.learner = SVMLearner(
                        gamma = 0.03,
                        C=0.2,
                        kernel = 'rbf'
                        )
        self.expHelper = ExperimentHelper(self.splitter, self.learner, '-iter-2')
        self.expHelper.learning_curve_exp()                

    def experiment_run_test_wine(self): 
        self.learner = SVMLearner(
                        gamma = 0.05,
                        C=0.5,
                        kernel = 'rbf'
                        ) 
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()  

    def experiment_run_test_bank(self): 
        self.learner = SVMLearner(
                        gamma = 0.03,
                        C=0.2,
                        kernel = 'rbf'
                        ) 
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test() 