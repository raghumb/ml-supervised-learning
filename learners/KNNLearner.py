from sklearn.neighbors import KNeighborsClassifier
import time	


class KNNLearner:

    def __init__(self, 
                verbose = False,
                n_neighbors = 5,
                weights = 'uniform',
                algorithm = 'brute',
                leaf_size = 30,
                p = 2,
                metric = 'minkowski',
                metric_params = None,
                n_jobs = None):
        self.verbose = verbose
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs      

        #Create KNN Classifier
        #n_neighbors=5
        self.classifier = KNeighborsClassifier(
                n_neighbors = self.n_neighbors,
                weights = self.weights,
                algorithm = self.algorithm,
                leaf_size = self.leaf_size,
                p = self.p,
                metric = self.metric,
                metric_params = self.metric_params,
                n_jobs = self.n_jobs
                )         
        pass


    def train(self, X_train, y_train):

        start = time.time()
        #Train the model using the training sets
        self.classifier.fit(X_train, y_train)
        end = time.time()
        diff = abs(end - start)
        print('train time taken for KNN: ' + str(diff))        

    def query(self, X_test):
        start = time.time()
        #Predict the response for test dataset
        y_pred = self.classifier.predict(X_test)
        end = time.time()
        diff = abs(end - start)
        print('test time taken for KNN: ' + str(diff))        

        return y_pred