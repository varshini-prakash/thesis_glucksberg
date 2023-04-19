from pydoc import doc
import pandas as pd
import numpy as np
import pickle as pk
import nibabel as nib
import random
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.utils import shuffle
import argparse
import sys
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import LeaveOneOut
import time
import os.path
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from ast import literal_eval
import glob

def cross_validaton_nested_svm(X, y):

    best_alphas, results, precision, recall, cm, y_tests, y_preds = [], [], [], [], [], [], []
        
    for i in range(20):
        
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.1,stratify = y,random_state=i)
        model = LogisticRegression(class_weight='balanced', random_state=i,solver = 'lbfgs', penalty = 'l2',max_iter=2000, n_jobs=-1)
        scoring = 'balanced_accuracy'
        ridge_params = {'C': [0.000001,0.00001,0.0001,0.0001,0.001,0.01, 0.1, 1, 10, 100,1000,10000]}
        clf = GridSearchCV(model, ridge_params, scoring=scoring, n_jobs=-1, cv=4)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        pca = PCA(n_components = 0.85)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

        clf.fit(X_train,y_train)
        best_alphas.append(clf.best_params_)
        print(clf.best_params_)
        y_pred = clf.predict(X_test)

        testScore = metrics.accuracy_score(y_test, y_pred)
        precision_score = metrics.precision_score(y_test, y_pred, average = 'weighted')
        recall_score = metrics.recall_score(y_test, y_pred, average = 'weighted')

        results.append(testScore)
        precision.append(precision_score)
        recall.append(recall_score)

        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print("Precision:",precision_score)
        print("Recall:",recall_score)

        #Confusion Matrix
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred, normalize = 'true')
        print(cnf_matrix)
        cm.append(cnf_matrix)
        y_tests.append(y_test)
        y_preds.append(y_pred)
        
    y_tests = np.array(y_tests)
    y_preds = np.array(y_preds)
    
    cm = np.array(cm)
    
    return y_tests,y_preds, cm

#Load betas

def load_betas(participant, beta_dir):
    
    
    with open(beta_dir + 'lt_betas/' + participant + '.pkl','rb') as f:
        lt_betas = pk.load(f)
    
    with open(beta_dir + 'lf_betas/' + participant + '.pkl','rb') as f:
        lf_betas = pk.load(f)
    
    with open(beta_dir + 'm_betas/' + participant + '.pkl','rb') as f:
        m_betas = pk.load(f)

    with open(beta_dir + 'sm_betas/' + participant + '.pkl','rb') as f:
        sm_betas = pk.load(f)
    
    all_betas = np.concatenate((lt_betas, lf_betas, m_betas, sm_betas), axis=0)
    return all_betas
    
# LT LF M SM
y_labels = [0]*80 + [1]*40 + [2]*20 + [3]*20
beta_dir = '/home/varshini/scratch/data/data_glucksberg/processed_data/betas/alphabetical/'
out_dir = '/home/varshini/scratch/src/thesis_glucksberg/output/clf_betas_19_apr/'

control_p = ['P054','P057','P064','P065','P067','P068','P072','P073','P075','P076','P080','P081']
ASD_p     = ['P050','P055','P056','P058','P059','P060','P066','P069','P070','P071','P078','P079']

all_participants = sorted(ASD_p + control_p)

for participant in all_participants:
    all_betas = load_betas(participant, beta_dir)
    y_tests,y_preds, cm = cross_validaton_nested_svm(all_betas, y_labels)
    
    with open(out_dir + participant + '_pred.pkl','wb') as f:
        pk.dump(y_preds, f)
        
    with open(out_dir + participant + '_test.pkl','wb') as f:
        pk.dump(y_tests, f)
        
    with open(out_dir + participant + '_cm.pkl','wb') as f:
        pk.dump(cm, f)
        
    print(participant)

