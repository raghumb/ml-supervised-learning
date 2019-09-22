import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import learning_curve
from plotter import plot_learning_curve
from plotter import plot_model_complexity_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from plotter import plot_roc_curve

#This code is derived from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#

class ExperimentHelper:
    def __init__(self, splitter, learner, prefix = None, title_pre = None):        
        self.splitter = splitter   
        self.learner = learner 
        self.title_pre = title_pre
        if prefix is None:
            self.prefix = None #self.splitter.reader.dataset
        else:
            self.prefix = self.splitter.reader.dataset + '-' + prefix 

    def learning_curve_exp(self):
        scoring = 'accuracy' #'neg_mean_squared_error'        
        """scores = cross_validate(self.learner.dt, 
                                self.splitter.X_train, 
                                self.splitter.y_train, 
                                cv = self.splitter.X_cv, 
                                scoring = scoring) """

        print('splitter train size '+ str(self.splitter.X_train.shape[0]))
        train_sizes, train_scores, validation_scores = learning_curve( 
                                                        self.learner.classifier,
                                                        self.splitter.X_train,
                                                        self.splitter.y_train,
                                                        train_sizes = self.splitter.learn_train_sizes,
                                                        cv = self.splitter.num_splits,
                                                        scoring = scoring,
                                                        shuffle = True,
                                                        random_state = 1)

        
        print(self.learner.__class__)
        learner_name = self.class_mapper(str(self.learner.__class__.__name__))
        print('Training scores:\n\n ', train_scores)
        print('\nValidation scores:\n\n', validation_scores)
        #print("Accuracy for " + str(learner.__class__)+": ", metrics.accuracy_score(y_test, y_pred))

        train_scores_mean = train_scores.mean(axis = 1)
        validation_scores_mean = validation_scores.mean(axis = 1)
        print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
        print('\n', '-' * 20) # separator
        print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))
        
        plot_learning_curve(train_sizes, train_scores_mean, validation_scores_mean, learner_name, self.prefix)  

    def model_complexity_exp(self, param_name, param_range):
        scoring = 'accuracy'
        print("in model complexity")
        print(self.learner.__class__.__name__)
        learner_name = self.class_mapper(str(self.learner.__class__.__name__))
        train_scores, validation_scores = validation_curve(
                                            self.learner.classifier,
                                            self.splitter.X_train,
                                            self.splitter.y_train,
                                            param_name,
                                            param_range,
                                            cv = self.splitter.num_splits,
                                            scoring = scoring
                                            )
        
        print('Training scores:\n\n ', train_scores)
        print('\nValidation scores:\n\n', validation_scores)
        #print("Accuracy for " + str(learner.__class__)+": ", metrics.accuracy_score(y_test, y_pred))

        train_mean = train_scores.mean(axis = 1)
        train_std = np.std(train_scores, axis=1)
        validation_mean = validation_scores.mean(axis = 1)
        validation_std = np.std(validation_scores, axis=1)
        #print('MC Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
        print('\n', '-' * 20) # separator
        #print('\n MC Mean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))
        
        plot_model_complexity_curve(param_name, param_range, train_mean, train_std, validation_mean, validation_std, learner_name, self.prefix, self.title_pre)
        

    def perform_grid_search(self, grid_param):
        print("Starting Grid search")
        scoring = 'accuracy'
        grid = GridSearchCV(estimator = self.learner.classifier,
                            param_grid = grid_param,
                            scoring = scoring,
                            cv = self.splitter.num_splits,
                            n_jobs = -1)

        grid.fit(self.splitter.X_train, self.splitter.y_train)
        print("End Grid search")
        learner_name = self.class_mapper(str(self.learner.__class__.__name__))
        best_params = grid.best_params_
        print('Best Params for ' + learner_name + ' : ') 
        print(best_params)  
        best_result = grid.best_score_
        print('Best score for ' + learner_name + ' : ') 
        print(best_result)
           
    
    def class_mapper(self, class_name):
        learner_name = 'None'
        if class_name == 'ANNLearner':
            learner_name = 'ANNLearner'
        if class_name == 'DTLearner':
            learner_name = 'DTLearner'   
        if class_name == 'BoostLearner':
            learner_name = 'BoostLearner'     
        if class_name == 'KNNLearner':
            learner_name = 'KNNLearner'           
        if class_name == 'SVMLearner':
            learner_name = 'SVMLearner'    
        if class_name == 'LinearSVMLearner':
            learner_name = 'LinearSVMLearner'                                  
        return learner_name  

    def experiment_run_test(self): 
        learner_name = self.class_mapper(str(self.learner.__class__.__name__))
        self.learner.train(self.splitter.X_train, self.splitter.y_train)
        y_pred = self.learner.query(self.splitter.X_test)
        print("Final Accuracy for " + str(self.learner.__class__)+": ", 
                        metrics.accuracy_score(self.splitter.y_test, y_pred))   
        print("Confusion matrix for " + str(self.learner.__class__)+": ", 
                        confusion_matrix(self.splitter.y_test, y_pred))        
        print("Recall score for " + str(self.learner.__class__)+": ", 
                        recall_score(self.splitter.y_test, y_pred))   
        print("Precision score for " + str(self.learner.__class__)+": ", 
                        precision_score(self.splitter.y_test, y_pred)) 
        print("f1 score for " + str(self.learner.__class__)+": ", 
                        f1_score(self.splitter.y_test, y_pred)) 

        fpr, tpr, thresholds = roc_curve(self.splitter.y_test, y_pred)
        

        #plot_roc_curve(fpr, tpr, thresholds, learner_name, self.prefix) 