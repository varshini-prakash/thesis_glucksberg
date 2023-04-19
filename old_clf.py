from pydoc import doc
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
#from gensim.models import KeyedVectors
import pickle as pk
import nibabel as nib
import pandas as pd
import numpy as np
import random
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
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


def cross_validaton_logreg(X, y):
    
    
    X, y = shuffle(X, y)
    best_alphas = []
    results = []
    precision = []
    recall = []
    cm = []
    y_tests, y_preds = [], []
    
    for i in range(5):
        test_indices = np.random.random_integers(0, 159, 32)
        train_indices = list(range(0,160))
        for index in sorted(test_indices, reverse = True):
            del train_indices[index]
        
        X_train, X_test = [], []
        y_train, y_test = [], []

        for index in train_indices:
            X_train.append(X[index])
            y_train.append(y[index])

        for index in test_indices:
            X_test.append(X[index])
            y_test.append(y[index])
        
        model = LogisticRegression(class_weight='balanced', random_state=i,solver = 'lbfgs', penalty = 'l2',max_iter=500, n_jobs=-1)
        scoring = 'accuracy'
        ridge_params = {'C': [0.001,0.01, 0.1, 1, 10, 100,1000]}
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

control_p = ['P054','P057','P064','P065','P067','P068','P072','P073','P075','P076','P080','P081']
ASD_p = ['P050','P055','P056','P059','P069','P070','P071','P078','P079']
all_participants = sorted(ASD_p + control_p)

beta_dir = '/home/varshini/projects/def-afyshe-ab/varshini/glucks/data/results_betas/betas_all/'
out_dir = 'clf_apr_18'
output_dir = '/home/varshini/projects/def-afyshe-ab/varshini/glucks/clf_jul/' + out_dir + '/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


corr_dir = output_dir
group = all_participants
reg_names = ['AG','IFG','ROI','dACC']
class_labels = [0]*80 + [1]*40 + [2]*20 + [3]*20
# LT LF M SM

for participant in group:

    print(participant)
    p_start = time.time()
    
    with open(beta_dir + 'all_betas_' + participant + '.pkl','rb') as f:
        all_betas = pk.load(f)
        
    #ROI
    X, y = all_betas[2], class_labels
    
    file_name_pred = output_dir + participant + '_' + reg_names[2] +'_pred.pkl'
    file_name_true = output_dir + participant + '_' + reg_names[2] +'_true.pkl'
    file_name_cm = output_dir + participant + '_' + reg_names[2] +'_cm.pkl'

    y_tests, y_preds, cm = cross_validaton_logreg(X,y)
    with open(file_name_pred,'wb') as f:
        pk.dump(y_preds, f)

    with open(file_name_true,'wb') as f:
        pk.dump(y_tests, f)

    with open(file_name_cm,'wb') as f:
        pk.dump(cm, f)
        



