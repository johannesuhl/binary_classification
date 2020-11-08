# -*- coding: utf-8 -*-
"""
Created on Sat Nov 07 22:44:26 2020

@author: Johannes H. Uhl, Department of Geography, Unversity of Colorado Boulder, USA.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

simulate_input = True
own_classification=False
use_diabetes_data=False
use_sonar_data=True

if simulate_input:
    
    #### get / generate data for classification #########################
    
    if own_classification:
        X, y = make_classification(n_samples=2500, n_features=100, flip_y=0.15, class_sep=0.6) # make classification with 15% label noise
    
    if use_diabetes_data:
        diabetes_datadf = pd.read_csv(r'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv',header=None)   
        y=diabetes_datadf[diabetes_datadf.columns[-1]].values
        X=diabetes_datadf[diabetes_datadf.columns[:-1]].values

    if use_sonar_data:
        sonar_datadf = pd.read_csv(r'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data',header=None)
        y=sonar_datadf[sonar_datadf.columns[-1]].values
        X=sonar_datadf[sonar_datadf.columns[:-1]].values
        y[y=='R']=0
        y[y=='M']=1
        y=y.astype(int)
        
    #### simulate the outputs of systematic hyperparameter tuning ########################################################################
    #### (here we vary number of iterations, and test the effects of train/test split proportions)
    #### and use a multilayer perceptron with fixed layer configuration
    #### in practice, this is certainly not meaningful - just to get some quick predictions that slightly differ.
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    y_tests=[]
    fscores=[]
    
    test_sizes = np.arange(0.01,0.991,0.1)
    max_iters=[50,100,500,1000,10000]
    y_tests=[]
    fscores=[]
    configs=[]
    i=0
    for test_size in test_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=test_size)
        for max_iter in max_iters:
            i+=1
            clf = MLPClassifier(hidden_layer_sizes=(100,32),max_iter=max_iter)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X)
            y_tests.append(y_pred)
            fscore=f1_score(y,y_pred)
            fscores.append(fscore)
            print(test_size,max_iters,fscore)
            configs.append(i)
    
    refclass_column=0
    ### add the reference classification, and a "dummy" F-score, since the reference labels should appear in the top row:
    y_tests.append(y)
    fscores.append(1.0) #dummy
    configs = np.append(configs,refclass_column) #dummy for the reference data
            
    y_tests_arr = np.array(y_tests)
    datadf=pd.DataFrame(np.rot90(y_tests_arr))
    datadf.columns=configs
    
#########################################################################

## in practice, the user specifies a pandas dataframe (datadf) here.
## with a ROW for each instance of the test dataset
## with a COLUMN for the prediced labels of each classification (e.g., using a range of hyperparameters)
## with each cell holding the binary labels (0 or 1)
## also, the dataframe needs a column <refclass_column> (to be declared by the user)
## holding the reference (validation) data labels.
## user also specifies a vector <fscores> holding the f-measure (or other preferred accuracy measure) for each classification
## the code below will then resort that data frame (along two axis)
    ## (a) based on the reference labels along the labels
    ## (b) based on the F-measure along the classifiers

## EXAMPLE:  
#datadf = pd.read_csv('hyperparameter_tuning_y_test.csv')
#refclass_column = 'reflabels'
#datadf[refclass_column] = pd.read_csv('reference_y_test.csv')

#########################################################################

### sort along first axis all labels by reference class labels
datadf=datadf.sort_values(by=refclass_column)

### sort along second axis by the f-score:
datadf_trans = datadf.transpose()
datadf_trans['fscore']=fscores
datadf_trans = datadf_trans.sort_values(by='fscore',ascending=False)

### plot
plotarr = datadf_trans.drop(labels=['fscore'],axis=1).values
fig, ax = plt.subplots(figsize=(16, 8))
ax.grid(False)
ax.set_yticks(np.arange(0, datadf_trans.fscore.values.shape[0], 1))
ax.set_yticklabels(['%1.3f' %x for x in datadf_trans.fscore.values],  size = 5)
ax.set_ylabel('F-score')       
plt.imshow(plotarr, interpolation='none', cmap='winter', aspect='auto')      
plt.show()
fig.savefig('multiple_binary_classifier_comparison.jpg', dpi=300, bbox_inches = 'tight')
plt.clf() 