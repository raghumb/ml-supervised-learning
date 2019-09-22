from experiments.ExperimentHelper import ExperimentHelper
from learners.ANNLearner import ANNLearner
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, make_scorer

class ANNExp:
    def __init__(self, reader, helper, splitter):
        self.reader = reader
        self.helper = helper
        self.learner = ANNLearner()
        self.splitter = splitter
        self.expHelper = ExperimentHelper(self.splitter, self.learner)

    def experiment(self):

        self.model_complexity_exp()
        self.model_complexity_exp_alpha1()
        self.model_complexity_exp_epoch()
        self.model_complexity_exp_epoch2()
        if(self.splitter.reader.dataset == 'Bank'):
            print('bank')
            self.learning_curve_iter2_bank()
            self.experiment_run_test_bank()            
        else:
            self.learning_curve_iter2_wine()
            self.experiment_run_test_wine()
        
        

        # Perform learning curve
        """self.expHelper.learning_curve_exp()
        self.model_complexity_exp()
        self.model_complexity_exp_alpha1()
        self.model_complexity_exp_alpha2()
        #self.exp_grid_search()"""
        #self.learning_curve_iter2()
        #self.exp_grid_search3()
        

        #self.learner.train(self.splitter.X_train, self.splitter.y_train)
        #y_pred = self.learner.query(self.splitter.X_test)
        """print("Final Accuracy for " + str(self.learner.__class__)+": ", 
                        metrics.accuracy_score(self.splitter.y_test, y_pred))"""

    def model_complexity_exp(self):
        #TODO should we create a new learner object??
        self.learner = ANNLearner()
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        param_range = np.array([(50,),(100,),(200,),(300,),(400,),(500,)])
        #param_range = np.array([100, 200,300,400, 500])
        self.expHelper.model_complexity_exp('hidden_layer_sizes', param_range)

    def model_complexity_exp11(self):
        #TODO should we create a new learner object??
        self.learner = ANNLearner()
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        param_range = np.array([(50,),(50,50,),(50,50,50,),(50,50,50,50,),(50,50,50,50,50,)])
        #param_range = np.array([100, 200,300,400, 500])
        self.expHelper.model_complexity_exp('hidden_layer_sizes', param_range)

    def model_complexity_exp_alpha21(self):
        #TODO should we create a new learner object??
        self.learner = ANNLearner(hidden_layer_sizes = (50,50,50,))
        self.expHelper = ExperimentHelper(self.splitter, self.learner, '2')
        #param_range = np.array([(100),(200,),(300,),(400,),(500,)])
        #param_range = np.array([0.0001, 0.001, 0.01, 0.1,1,5])
        param_range = np.array([0.0001, 0.001,0.002,0.003,0.005,0.008])
        self.expHelper.model_complexity_exp('alpha', param_range)         

    def model_complexity_exp_epoch(self):
        #TODO should we create a new learner object??
        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.01,
            hidden_layer_sizes = (50,50,50,),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = True
        )
        self.expHelper = ExperimentHelper(self.splitter, self.learner, '2')
        #param_range = np.array([(100),(200,),(300,),(400,),(500,)])
        #param_range = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
        param_range = np.array([1, 10, 50, 100, 200, 500])
        self.expHelper.model_complexity_exp('max_iter', param_range) 

    def model_complexity_exp_epoch2(self):
        #TODO should we create a new learner object??
        self.learner = ANNLearner(hidden_layer_sizes = (300,), 
        alpha = 0.0001, early_stopping = False)
        self.expHelper = ExperimentHelper(self.splitter, self.learner, '3')
        #param_range = np.array([(100),(200,),(300,),(400,),(500,)])
        #param_range = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
        param_range = np.array([1, 10, 50, 100, 200, 500, 1000])
        self.expHelper.model_complexity_exp('max_iter', param_range) 

    def model_complexity_exp_epoch3(self):
        #TODO should we create a new learner object??
        self.learner = ANNLearner(hidden_layer_sizes = (200,100), 
        alpha = 0.008, early_stopping = True, solver='sgd', activation='tanh' )
        self.expHelper = ExperimentHelper(self.splitter, self.learner, '4')
        #param_range = np.array([(100),(200,),(300,),(400,),(500,)])
        #param_range = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
        param_range = np.array([1, 10, 50, 100, 200, 500, 1000])
        self.expHelper.model_complexity_exp('max_iter', param_range)               


    def model_complexity_exp_alpha1(self):
        #TODO should we create a new learner object??
        self.learner = ANNLearner(hidden_layer_sizes = (300,))
        self.expHelper = ExperimentHelper(self.splitter, self.learner, '1')
        #param_range = np.array([(100),(200,),(300,),(400,),(500,)])
        #param_range = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
        param_range = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
        self.expHelper.model_complexity_exp('alpha', param_range) 

        self.learner = ANNLearner(hidden_layer_sizes = (390,))
        self.expHelper = ExperimentHelper(self.splitter, self.learner, '2')
        #param_range = np.array([(100),(200,),(300,),(400,),(500,)])
        param_range = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        #param_range = np.array([0.0001, 0.0005, 0.001, 0.005, 0.008])
        self.expHelper.model_complexity_exp('alpha', param_range)   

    def model_complexity_exp_alpha2(self):
        #TODO should we create a new learner object??
        self.learner = ANNLearner()
        self.expHelper = ExperimentHelper(self.splitter, self.learner, '2')
        #param_range = np.array([(100),(200,),(300,),(400,),(500,)])
        param_range = np.array([0.0001, 0.001, 0.01, 0.1,1,5])
        self.expHelper.model_complexity_exp('alpha', param_range)   

    def exp_grid_search2(self):
        grid_param = {
            'hidden_layer_sizes': [(50,50,50), (50,50,50,50),(50,50,50,50,50)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.1, 0.2, 0.3, 0.4, 0.05],
            'learning_rate': ['constant'],
            'momentum' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'max_iter': [200, 300, 500, 700, 1000, 1500, 2000, 2500],
        }
        self.expHelper.perform_grid_search(grid_param)  

    def exp_grid_search(self):
        grid_param = {
            'hidden_layer_sizes': [(50,),(20,20,20)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.05, 0.1,0.2,0.4,0.5,0.6],
            'learning_rate': ['constant'],
            'learning_rate_init': [0.001, 0.002, 0.004],
            'shuffle': [True, False],
            'early_stopping': [True, False],
            'beta_1': [0.9, 0.5],
            'beta_2': [0.999, 0.5]

        }
        self.expHelper.perform_grid_search(grid_param) 

    def exp_grid_search3(self):
        grid_param = {
            'hidden_layer_sizes': [(50,50), (100,100), (200,200),(300,300),(400,400),(500,500)],
            'activation': ['relu'],
            'solver': ['sgd'],
            'alpha': [0.0001, 0.05, 0.1,0.2,0.4,0.5,0.6],
            'learning_rate': ['constant','adaptive'],
        }
        self.expHelper.perform_grid_search(grid_param)    

    def exp_grid_search4(self):
        grid_param = {
            'hidden_layer_sizes': [(50,50,50), (100,100,100), (200,200,200),(300,300,100),(400,400,100),(500,500,100)],
            'activation': ['relu'],
            'solver': ['sgd'],
            'alpha': [0.2],
            'learning_rate': ['constant']
        }
        self.expHelper.perform_grid_search(grid_param)                                            

    def learning_curve_iter21_wine(self):
        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.008,
            hidden_layer_sizes = (100,),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        self.expHelper = ExperimentHelper(self.splitter, self.learner, '-iter-2')
        self.expHelper.learning_curve_exp()

    def learning_curve_iter2_wine(self):
        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.0001,
            hidden_layer_sizes = (50,50,50,50,),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        self.expHelper = ExperimentHelper(self.splitter, self.learner, '-iter-3')
        self.expHelper.learning_curve_exp()

    def learning_curve_iter2_bank(self):
        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.01,
            hidden_layer_sizes = (50,50,50,),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        self.expHelper = ExperimentHelper(self.splitter, self.learner, '-iter-3')
        self.expHelper.learning_curve_exp()       

    def learning_curve_iter21_bank(self):
        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.01,
            hidden_layer_sizes = (300,),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        self.expHelper = ExperimentHelper(self.splitter, self.learner, '-iter-2')
        self.expHelper.learning_curve_exp()

    def experiment_run_test_wine2(self): 
        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.008,
            hidden_layer_sizes = (100,),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = True
        )
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test() 


    def experiment_run_test_bank2(self): 
        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.7,
            hidden_layer_sizes = (390,),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()  

    def experiment_run_test_bank(self): 
        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.01,
            hidden_layer_sizes = (50,50,50,),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()         


    def experiment_run_test_wine(self): 
        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.0001,
            hidden_layer_sizes = (50,50,50,50,),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test() 
    
    def experiment_run_test_bank_iter2(self):
        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.06,
            hidden_layer_sizes = (200, 200, 200, 200, 200, 200, 200),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = True,
            max_iter = 600,
            momentum = 0.4
        )
        print('100, 100, 100, 100, 100, 100,50 alpha 0.3')
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test() 

        """ self.learner = ANNLearner(
            activation = 'tanh',
            alpha = 0.2,
            hidden_layer_sizes = (50, 50, 50, 50, 50, 50, 50, 50),
            learning_rate = 'constant',
            solver = 'sgd',
            early_stopping = False
        )
        print('100, 100, 100, 100, 100, 50,50, 50 alpha 0.3')
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()    

        self.learner = ANNLearner(
            activation = 'tanh',
            alpha = 0.6,
            hidden_layer_sizes = (100, 100, 100, 100, 100, 50, 50, 50, 50),
            learning_rate = 'constant',
            solver = 'sgd',
            early_stopping = False
        )
        print('100, 100, 100, 100, 100, 50,50, 50, 50 alpha 0.3')
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()  """            
    
    
    def experiment_run_test_bank_iter(self): 
        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.7,
            hidden_layer_sizes = (390, 100),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        print('390,100')
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()
    
        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.7,
            hidden_layer_sizes = (200, 200, 200),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        print('200, 200, 200')
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()

        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.7,
            hidden_layer_sizes = (400, 400, 400),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        print('400, 400, 400')
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()

        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.7,
            hidden_layer_sizes = (500, 500, 500),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        print('500, 500, 500')
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()        

        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.7,
            hidden_layer_sizes = (200, 300, 400),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        print('200, 300, 400')
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()         


        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.7,
            hidden_layer_sizes = (390,100),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        print('390,100')
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()
    
        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.3,
            hidden_layer_sizes = (200, 200, 200),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        print('200, 200, 200 alpha 0.3')
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()

        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.3,
            hidden_layer_sizes = (400, 400, 400),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        print('400, 400, 400  alpha 0.3')
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()

        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.3,
            hidden_layer_sizes = (500, 500, 500),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        print('500, 500, 500  alpha 0.3')
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()        

        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.3,
            hidden_layer_sizes = (200, 300, 400),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        print('200, 300, 400  alpha 0.3')
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test() 

        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.3,
            hidden_layer_sizes = (100, 100, 100, 100,),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        print('100, 100, 100, 100,  alpha 0.3')
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()                  

        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.3,
            hidden_layer_sizes = (100, 100, 100, 100, 100),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        print('100, 100, 100, 100, 100, alpha 0.3')
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test() 

        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.3,
            hidden_layer_sizes = (100, 100, 100, 100, 100, 50),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        print('100, 100, 100, 100, 100, 50, alpha 0.3')
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()   

        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.3,
            hidden_layer_sizes = (50, 50, 50, 50, 50, 50),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        print('50, 50, 50, 50, 50, 50, alpha 0.3')
        self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.expHelper.experiment_run_test()                      