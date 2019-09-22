# ml-supervised-learning

1. Setup Mini Conda:

Download Conda:
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

Install conda:
sh Miniconda3-latest-Linux-x86_64.sh

2. Create  virtual env (Use the environment.yml):
conda env create --file environment.yml

3. Activate environment:
conda activate ml-raghu

4. pip install xgboost

5. Run the experiments using: This will run all the experiments for both the datasets(wine and bank):
PYTHONPATH=../:. python -W ignore  experiment.py



References:
1. https://stackoverflow.com/questions/49428469/pruning-decision-trees

2. API design for machine learning software: experiences from the scikit-learn project, Buitinck et al., 2013. (Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#)

3. API design for machine learning software: experiences from the scikit-learn project, Buitinck et al., 2013. (Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

4. API design for machine learning software: experiences from the scikit-learn project, Buitinck et al., 2013. (Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#)

5. API design for machine learning software: experiences from the scikit-learn project, Buitinck et al., 2013. (Retrieved from https://scikit-learn.org/stable/modules/svm.html#)

6. API design for machine learning software: experiences from the scikit-learn project, Buitinck et al., 2013. (Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#)

7. API design for machine learning software: experiences from the scikit-learn project, Buitinck et al., 2013. (Retrieved from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#)

8. API design for machine learning software: experiences from the scikit-learn project, Buitinck et al., 2013. (Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#)

9. Brownlee, Jason. (2019, August 21). A Gentle Introduction to XG Boost for Applied Machine Learning. (Retrieved from https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)

10.	P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009. (Retrieved from https://archive.ics.uci.edu/ml/datasets/wine+quality)
11.	[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014 (Retrieved from https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
