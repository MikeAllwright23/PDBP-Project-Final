from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
# roc curve and auc score
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor as GBM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#from sklearn.linear_model import GradientBoostingClassifier 
from sklearn.mixture import GMM
from boruta import BorutaPy

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn import cross_validation
from sklearn.model_selection import train_test_split

from sklearn import ensemble


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import xgboost as xgb
#from glmnet import LogitNet
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB

from sklearn import linear_model

from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA

from sklearn.preprocessing import StandardScaler

import random

import umap
import hdbscan

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
   # warnings.simplefilter("default")
from sklearn.model_selection import RepeatedKFold

def ncvmodel(X,y,featsel,names,classifiers,y_var="PD",name="test",reps=100,umap_c=30,um_neigh=20,pca_comp=20,n_splits=3):

    mod=np.array([])
    i=0
    random_state = 12883823
    fs=featsel
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=reps, random_state=random_state)

    if fs=="UMap":
        import umap
        n_neighbors=um_neigh
        reducer    = umap.UMAP(n_neighbors=n_neighbors,n_components=umap_c)
        X_Um = reducer.fit_transform(X_used)
        X_m=X_Um
    if fs=="auto":
        X_m=X_auto
        
    if fs=="none":
        X_m=X
        
    models=np.array([])
    auc_vals=np.array([])
    for train_index, test_index in rkf.split(X_m):
        X_train, X_test = X_m[train_index], X_m[test_index]
        y_train, y_test = y[train_index], y[test_index]
  
        mod=np.array([])
        inc=np.array([])
        pbs=0
        ct=0
        i=i+1
       
        for j in range(len(names)):

            model=classifiers[j]
            modname=names[j]
            model.fit(X_train,y_train)
                
            probs = model.predict_proba(X_test)
            probs = probs[:, 1]
            y_vals=np.where(y_test==y_var, 1, 0)
            auc = roc_auc_score(y_vals, probs)
            inc=np.append(inc,auc)
            mod=np.append(mod,modname)
           
            if names[j]=="Naive Bayes" or names[j]=="Log Reg" or names[j]=="KNN":
                pbs=pbs+probs
                ct=ct+1
            #ensemble voting
        #ensemble prediction
        probs_ens=pbs/ct
        y_vals=np.where(y_test==y_var, 1, 0)
        auc_ens = roc_auc_score(y_vals, probs_ens)
        mod_ens="ensemble many"
         
        auc_vals=np.append(auc_vals,inc)
        auc_vals=np.append(auc_vals,auc_ens)
        
        models=np.append(models,mod)
        models=np.append(models,mod_ens)

    testing_data = pd.DataFrame({'Model':models,'auc':auc_vals})
    testing_data=testing_data.sort_values(by='Model')
    mod_sum=pd.DataFrame(testing_data.groupby('Model').agg({'auc':['min','max','mean','std']})).reset_index()
    mod_sum.columns=['Model','AUC min','AUCmax','AUC Performance Metric','AUCstd']
    
    testing_data.to_csv("%s%s" % (path,str(name)+'.csv'))
    
    with PdfPages("%s%s" % (path,str(name)+'.pdf')) as export_pdf:
        fig = plt.figure(figsize=(30,10))
        ax = sns.boxplot(x="Model",y="auc", data=testing_data)
        ax = sns.swarmplot(x="Model",y="auc", data=testing_data, color=".25")
        export_pdf.savefig()       
        plt.show()
        plt.close()
     
        y=mod_sum['AUC Performance Metric']
        x=mod_sum['Model']
        
        barplot(x,y,mod_sum) 
        export_pdf.savefig()  
        plt.show()
        plt.close()
           
        y=mod_sum['AUCstd']
        barplot(x,y,mod_sum) 
        export_pdf.savefig()  
        plt.show()
        plt.close()
    return testing_data,mod_sum