from learners.DTLearner import DTLearner
from BankDataReader import BankDataReader
from WineDataReader import WineDataReader
from experiments.DTExp import DTExp
from experiments.ANNExp import ANNExp
from experiments.KNNExp import KNNExp
from experiments.SVMExp import SVMExp
from experiments.BoostingExp import BoostingExp
from DataSplitter import DataSplitter

def experiment_run(exp, test_run = False, 
                        grid_search = False, 
                        learning_curve_2 = False,
                        dataset = 'bank'):


    for e in exp:
        if test_run == True:
            print('Test Run')
            if dataset == 'bank':
                e.experiment_run_test_bank() 
            else:
                e.experiment_run_test_wine() 
        elif grid_search == True:
            e.exp_grid_search() 
        elif learning_curve_2 == True:
            if dataset == 'bank':
                e.learning_curve_iter2_bank() 
            else:
                e.learning_curve_iter2_wine() 
        else:      
            e.experiment()


def save_bank_data():
    reader = BankDataReader()
    reader.read_process_data()       

def experiment_invoke(dataset = 'bank'):
    if dataset == 'wine':
        reader = WineDataReader()
    else:
        reader = BankDataReader()
    
    
    h = None
    ds = DataSplitter(reader)
    ds.read_split_data()

    dtl = DTExp(reader, h, ds)
    ann = ANNExp(reader, h, ds)
    boost = BoostingExp(reader, h, ds) 
    knn = KNNExp(reader, h, ds)
    svm = SVMExp(reader, h, ds)
    
    #exp = [dtl, knn, svm]
    exp = [knn, dtl, svm, boost, ann]
    #exp = [ann]
    experiment_run(exp, False, False, False) 



    

experiment_invoke(dataset = 'wine')
experiment_invoke(dataset = 'bank')