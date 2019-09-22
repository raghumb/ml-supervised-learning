from sklearn.ensemble import GradientBoostingClassifier


class BoostLearner:

    def __init__(self, 
                verbose = False,
                base_estimator = None,
                n_estimators = 50,
                learning_rate = 1.0,
                algorithm = 'SAMME.R',
                random_state = 1):
        self.verbose = verbose
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state 

        #Create Boost Classifier
        self.classifier = GradientBoostingClassifier(
                        algorithm = self.algorithm,
                        base_estimator = self.base_estimator,
                        learning_rate = self.learning_rate,
                        n_estimators = self.n_estimators,
                        random_state = self.random_state)     
        pass


    def train(self, X_train, y_train):


        #Train the model using the training sets
        self.classifier.fit(X_train, y_train)

    def query(self, X_test):
        #Predict the response for test dataset
        y_pred = self.classifier.predict(X_test)

        return y_pred