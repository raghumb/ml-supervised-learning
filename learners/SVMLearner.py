from sklearn import svm
import time	
#This code is derived from https://scikit-learn.org/stable/modules/svm.html
class SVMLearner:

    def __init__(self, 
                C = 1.0,
                kernel = 'rbf',
                degree = 3,
                gamma = 'auto_deprecated',
                coef0 = 0.0,
                shrinking = True,
                probability = False,
                tol = 0.001,
                cache_size = 200,
                class_weight = None,
                verbose = False,
                decision_function_shape = 'ovr',
                random_state = 1,
                max_iter = -1):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state
        self.max_iter = max_iter

        #Create a svm Classifier
        self.classifier = svm.SVC(
            C = self.C,
            kernel = self.kernel,
            degree = self.degree,
            gamma = self.gamma,
            coef0 = self.coef0,
            shrinking = self.shrinking,
            probability = self.probability,
            tol = self.tol,
            cache_size = self.cache_size,
            class_weight = None, #self.class_weight,
            verbose = self.verbose,            
            decision_function_shape = self.decision_function_shape,
            random_state = self.random_state,
            max_iter = self.max_iter
            )

        pass


    def train(self, X_train, y_train):

        start = time.time()
        #Train the model using the training sets
        self.classifier.fit(X_train, y_train)
        end = time.time()
        diff = abs(end - start)
        print('train time taken for SVM: ' + str(diff))        

    def query(self, X_test):
        start = time.time()
        #Predict the response for test dataset
        y_pred = self.classifier.predict(X_test)
        end = time.time()
        diff = abs(end - start)
        print('test time taken for SVM: ' + str(diff))          

        return y_pred