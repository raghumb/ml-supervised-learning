from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import time	


class BoostLearner:

    def __init__(self, 
                max_depth = 3,
                learning_rate = 0.1,
                n_estimators = 100,
                verbosity = 1,
                silent = None,
                objective = 'binary:logistic',
                booster = 'gbtree',
                n_jobs = 1,
                nthread = None,
                gamma = 0,
                min_child_weight = 1,
                max_delta_step = 0,
                subsample = 1,
                colsample_bytree = 1,
                colsample_bylevel = 1,
                colsample_bynode = 1,
                reg_alpha = 0,
                reg_lambda = 1,
                scale_pos_weight = 1,
                base_score = 0.5,
                random_state = 1,
                seed = None,
                missing = None):


        #Create Boost Classifier
        self.classifier = xgb.XGBClassifier(
                        max_depth = max_depth,
                        learning_rate = learning_rate,
                        n_estimators = n_estimators,
                        verbosity = verbosity,
                        silent = verbosity,
                        objective = objective,
                        booster = booster,
                        n_jobs = n_jobs,
                        nthread = nthread,
                        gamma = gamma,
                        min_child_weight = min_child_weight,
                        max_delta_step = max_delta_step,
                        subsample = subsample,
                        colsample_bytree = colsample_bytree,
                        colsample_bylevel = colsample_bylevel,
                        colsample_bynode = colsample_bynode,
                        reg_alpha = reg_alpha,
                        reg_lambda = reg_lambda,
                        scale_pos_weight = scale_pos_weight,
                        base_score = base_score,
                        random_state = random_state,
                        seed = seed,
                        missing = missing)     
        pass


    def train(self, X_train, y_train):

        start = time.time()
        #Train the model using the training sets
        self.classifier.fit(X_train, y_train)
        end = time.time()
        diff = abs(end - start)
        print('train time taken for Boost: ' + str(diff))

    def query(self, X_test):
        start = time.time()
        #Predict the response for test dataset
        y_pred = self.classifier.predict(X_test)
        end = time.time()
        diff = abs(end - start)
        print('test time taken for Boost: ' + str(diff))

        return y_pred