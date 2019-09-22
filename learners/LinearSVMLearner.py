from sklearn.svm import LinearSVC
import time	

class LinearSVMLearner:

    def __init__(self, 
                penalty='l2',
                loss='squared_hinge',
                dual=True,
                tol=0.0001,
                C=1.0,
                multi_class='ovr',
                fit_intercept=True,
                intercept_scaling=1,
                class_weight=None,
                verbose=0,
                random_state=None,
                max_iter=1000):


        #Create a svm Classifier
        self.classifier = LinearSVC(
                penalty = penalty,
                loss = loss,
                dual = dual,
                tol = tol,
                C = C,
                multi_class = multi_class,
                fit_intercept = fit_intercept,
                intercept_scaling = intercept_scaling,
                class_weight = class_weight,
                verbose = verbose,
                random_state = random_state,
                max_iter = max_iter
            )

        pass


    def train(self, X_train, y_train):

        start = time.time()
        #Train the model using the training sets
        self.classifier.fit(X_train, y_train)
        end = time.time()
        diff = abs(end - start)
        print('train time taken for Linear SVM: ' + str(diff))       

    def query(self, X_test):
        start = time.time()
        #Predict the response for test dataset
        y_pred = self.classifier.predict(X_test)
        end = time.time()
        diff = abs(end - start)
        print('test time taken for Linear SVM: ' + str(diff))        

        return y_pred