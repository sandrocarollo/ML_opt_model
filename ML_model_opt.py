#!/usr/bin/env python

### IMPORT
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')
from scipy.stats import spearmanr
from scipy import interp
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, mean_squared_error, precision_score, jaccard_score, fowlkes_mallows_score, roc_auc_score, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import sklearn
import pickle
from datetime import timedelta



### FUNCTIONS
#split name
def splitname(x):
    return x.split('_')[0]

#CV split
def defineSplits(X,ycateg,random_state):
    from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
    # CV based on RCB categories
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(random_state))
    splits = []
    for (tr,ts) in cv.split(X, ycateg):
        splits.append((tr,ts))
    return splits

#Select K best features
class SelectAtMostKBest(SelectKBest):
    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            # set k to "all" (skip feature selection), if less than k features are available
            self.k = "all"
            
            
#Removing correlated Features
class DropCollinear(BaseEstimator, TransformerMixin):
    def __init__(self, thresh):
        self.uncorr_columns = None
        self.thresh = thresh

    def fit(self, X, y):
        cols_to_drop = []

        # Find variables to remove
        X_corr = X.corr()
        large_corrs = X_corr>self.thresh
        indices = np.argwhere(large_corrs.values)
        indices_nodiag = np.array([[m,n] for [m,n] in indices if m!=n])

        if indices_nodiag.size>0:
            indices_nodiag_lowfirst = np.sort(indices_nodiag, axis=1)
            correlated_pairs = np.unique(indices_nodiag_lowfirst, axis=0)
            resp_corrs = np.array([[np.abs(spearmanr(X.iloc[:,m], y).correlation), np.abs(spearmanr(X.iloc[:,n], y).correlation)] for [m,n] in correlated_pairs])
            element_to_drop = np.argmin(resp_corrs, axis=1)
            list_to_drop = np.unique(correlated_pairs[range(element_to_drop.shape[0]),element_to_drop])
            cols_to_drop = X.columns.values[list_to_drop]

        cols_to_keep = [c for c in X.columns.values if c not in cols_to_drop]
        self.uncorr_columns = cols_to_keep

        return self

    def transform(self, X):
        return X[self.uncorr_columns]

    def get_params(self, deep=False):
        return {'thresh': self.thresh}

#Optimisation parameter for SVM classifier
def optimise_SVC_featsel(X, y, cut, cv=5, label='Response', prefix='someresponse'):
    # Pipeline components
    scaler = StandardScaler()
    kbest = SelectAtMostKBest(score_func=f_classif)
    dropcoll = DropCollinear(cut)
    svc = SVC(random_state=1, max_iter=-1, probability=True)
    pipe = Pipeline(steps=[('dropcoll', dropcoll), ('scaler', scaler), ('kbest', kbest), ('svc', svc)])

    param_grid = { 'kbest__k': np.arange(2,X.shape[1],1),
                    'svc__kernel': ['rbf','sigmoid','linear'],
                    'svc__gamma': np.logspace(-9,-2,60),
                    'svc__C': np.logspace(-3,3,60)}

    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, iid=False, cv=cv, scoring='roc_auc',return_train_score=True, n_jobs=-1, verbose=0, n_iter=1000, random_state=0)
    search.fit(X,y)

    return search

#Optimisation parameter for Logistic Regression classifier
def optimise_logres_featsel(X, y, cut, cv=5, label='Response', prefix='someresponse', metric='roc_auc'):
    # Pipeline components
    scaler = StandardScaler()
    kbest = SelectAtMostKBest(score_func=f_classif)
    dropcoll = DropCollinear(cut)
    logres = LogisticRegression(random_state=1, penalty='elasticnet', solver='saga', max_iter=10000, n_jobs=-1, class_weight=True)
    pipe = Pipeline(steps=[('dropcoll', dropcoll), ('scaler', scaler), ('kbest', kbest), ('logres', logres)])
    # Parameter ranges
    param_grid = { 'kbest__k': np.arange(2,X.shape[1],1),
                    'logres__C': np.logspace(-3,3,30),
                    'logres__l1_ratio': np.arange(0.1,1.1,0.1) }
    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, iid=False, cv=cv, scoring=metric, return_train_score=True, n_jobs=-1, verbose=0, n_iter=1000, random_state=0)
    search.fit(X,y)

    return search


#Optimisation parameter for Random Forest classifier
def optimise_rf_featsel(X, y, cut, cv=5, label='Response', prefix='someresponse'):
    # Pipeline components
    scaler = StandardScaler()
    kbest = SelectAtMostKBest(score_func=f_classif)
    dropcoll = DropCollinear(cut)
    rf = RandomForestClassifier(random_state=1)
    pipe = Pipeline(steps=[('dropcoll', dropcoll), ('scaler', scaler), ('kbest', kbest), ('rf', rf)])
    # Parameter ranges
    param_grid = { 'kbest__k': range(1,X.shape[1]),
                    "rf__max_depth": [3, None],
                    "rf__n_estimators": [5, 10, 25, 50, 100],
                    "rf__max_features": [0.05, 0.1, 0.2, 0.5, 0.7],
                    "rf__min_samples_split": [2, 3, 6, 10, 12, 15]
                    }
    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, iid=False, cv=cv, scoring='roc_auc',return_train_score=True, n_jobs=-1, verbose=0,n_iter=1000, random_state=1)
    search.fit(X,y)

    return search





###DATASET READ
#Radiogenomics dataset + Meta info and Klinik dataset
Dataset_RG = pd.read_csv('./input/NSCLC_features_results_RG.csv',sep=',', header=0)
Dataset_Meta_RG = pd.read_csv('./input/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv',sep=',', header=0)
Dataset_kl = pd.read_csv('./input/NSCLC_features_results_kl.csv',sep=',', header=0)
Dataset_Meta_kl = pd.read_excel("./input/NSCLC_TLS_Projekt_09_20_AL_anonym.xlsx", sheet_name='NSCLC_TLS_Projekt_09_20')

## RG dataset
#removing non needed column
cols = [*range(1, 39, 1)]
dataset_RG=Dataset_RG.drop(Dataset_RG.columns[cols],axis=1)
#removing part of the Image name to sort properly
dataset_RG['Image'] = dataset_RG['Image'].apply(splitname)
#Sort by Image name
dataset_RG=dataset_RG.sort_values(by=['Image'])
#get image name to compare it with metadata info
col_one_arr = dataset_RG['Image'].to_numpy()
#don't need Image column anymore -> removing later after join
# datatest_RG=dataset_RG.drop('Image',axis=1)

#METADATA RG
#collect metadata and relapse status info
Dataset_Meta_RG=Dataset_Meta_RG.loc[:,["Case ID","Pathological T stage","Pathological N stage","Pathological M stage","Recurrence","Age at Histological Diagnosis"]]
Dataset_Meta_RG=Dataset_Meta_RG.loc[Dataset_Meta_RG['Case ID'].isin(col_one_arr)]
##Codification
Dataset_Meta_RG.loc[Dataset_Meta_RG["Recurrence"] == "yes", "Recurrence"] = 1
Dataset_Meta_RG.loc[Dataset_Meta_RG["Recurrence"] == "no", "Recurrence"] = 0
# pM
Dataset_Meta_RG.loc[Dataset_Meta_RG["Pathological M stage"] == "M0", "Pathological M stage"] = 0
Dataset_Meta_RG.loc[Dataset_Meta_RG["Pathological M stage"] == "M1", "Pathological M stage"] = 1
Dataset_Meta_RG.loc[Dataset_Meta_RG["Pathological M stage"] == "M1a", "Pathological M stage"] = 1
Dataset_Meta_RG.loc[Dataset_Meta_RG["Pathological M stage"] == "M1b", "Pathological M stage"] = 1

# pN
Dataset_Meta_RG.loc[Dataset_Meta_RG["Pathological N stage"] == "N0", "Pathological N stage"] = 0
Dataset_Meta_RG.loc[Dataset_Meta_RG["Pathological N stage"] == "N1", "Pathological N stage"] = 1
Dataset_Meta_RG.loc[Dataset_Meta_RG["Pathological N stage"] == "N2", "Pathological N stage"] = 2
# pT
Dataset_Meta_RG.loc[Dataset_Meta_RG["Pathological T stage"] == "Tis", "Pathological T stage"] = 0
Dataset_Meta_RG.loc[Dataset_Meta_RG["Pathological T stage"] == "T1a", "Pathological T stage"] = 1
Dataset_Meta_RG.loc[Dataset_Meta_RG["Pathological T stage"] == "T1b", "Pathological T stage"] = 2
Dataset_Meta_RG.loc[Dataset_Meta_RG["Pathological T stage"] == "T2a", "Pathological T stage"] = 3
Dataset_Meta_RG.loc[Dataset_Meta_RG["Pathological T stage"] == "T2b", "Pathological T stage"] = 4
Dataset_Meta_RG.loc[Dataset_Meta_RG["Pathological T stage"] == "T3", "Pathological T stage"] = 5
Dataset_Meta_RG.loc[Dataset_Meta_RG["Pathological T stage"] == "T4", "Pathological T stage"] = 6
#rename column for join
Dataset_Meta_RG=Dataset_Meta_RG.rename(columns={"Case ID": "Image"})
Dataset_Meta_RG=Dataset_Meta_RG.rename(columns={"Pathological T stage": "pT"})
Dataset_Meta_RG=Dataset_Meta_RG.rename(columns={"Pathological N stage": "pN"})
Dataset_Meta_RG=Dataset_Meta_RG.rename(columns={"Pathological M stage": "pM"})
Dataset_Meta_RG=Dataset_Meta_RG.rename(columns={"Age at Histological Diagnosis": "Age_Diag"})
#relapse target vector
target_RG=Dataset_Meta_RG.loc[:,"Recurrence"].values
target_RG=target_RG.astype(int)
#Wavelet and non-Wavelet feature dataset
# NON CLINICAL DATA included
# datatest_RG_W=datatest_RG  #all wavelet
# datatest_RG_NW=datatest_RG.iloc[:,0:107] #no-wavelet
# CLINICAL DATA included
datatest_RG_W=pd.merge(dataset_RG, Dataset_Meta_RG, on="Image")  #all wavelet
datatest_RG_NW=dataset_RG.iloc[:,0:107] #no-wavelet
datatest_RG_NW=pd.merge(datatest_RG_NW, Dataset_Meta_RG, on="Image")  #no wavelet
datatest_RG_W=datatest_RG_W.drop('Image',axis=1)
datatest_RG_W=datatest_RG_W.drop('Recurrence',axis=1)
datatest_RG_NW=datatest_RG_NW.drop('Image',axis=1)
datatest_RG_NW=datatest_RG_NW.drop('Recurrence',axis=1)

## Klinik dataset 
#cleaning and preparation
#keepin first column for sorting
cols = [*range(1, 39, 1)]
Dataset_kl=Dataset_kl.drop(Dataset_kl.columns[cols],axis=1)
#keeping only the number in the name for sort
Dataset_kl['Image'] = Dataset_kl['Image'].apply(splitname)
#converting string to number
Dataset_kl.Image = Dataset_kl.Image.astype(int)
#actual sort
Dataset_kl=Dataset_kl.sort_values(by=['Image'])
# non wavelet
datatest_kl_NW=Dataset_kl.iloc[:,0:108]
# wavelet
datatest_kl_W=Dataset_kl

# Dataset_kl=Dataset_kl.iloc[:,39::]
# datatest_kl_W=Dataset_kl #all wavelet
# datatest_kl_NW=Dataset_kl.iloc[:,0:107] #no-wavelet

#METADATA kl
# had to remove some blank rows.....
Dataset_Meta_kl=Dataset_Meta_kl.loc[0:74,:]
#converting string to int
Dataset_Meta_kl.Nummer = Dataset_Meta_kl.Nummer.astype(int)
#sort them
Dataset_Meta_kl=Dataset_Meta_kl.sort_values(by=['Nummer'])
#getting only the intersection between the two datasets
col_meta_kl = Dataset_Meta_kl['Nummer'].to_numpy()
col_data_kl = datatest_kl_NW['Image'].to_numpy()
Img_intersec=np.intersect1d(col_data_kl,col_meta_kl)
Dataset_Meta_kl=Dataset_Meta_kl.loc[Dataset_Meta_kl['Nummer'].isin(Img_intersec)]
#Klinik dataset target vector
#need to remove a patient
Dataset_Meta_kl = Dataset_Meta_kl[Dataset_Meta_kl.Rezidiv != 2]
Dataset_Meta_kl = Dataset_Meta_kl[Dataset_Meta_kl.Nummer != 128] #extra to remove cause of pTNM missing
#48 is the image with no relapse info, need to remove from the Radiomics dataset
datatest_kl_W = datatest_kl_W[datatest_kl_W.Image != 48]
datatest_kl_NW = datatest_kl_NW[datatest_kl_NW.Image != 48]
datatest_kl_W = datatest_kl_W[datatest_kl_W.Image != 128]
datatest_kl_NW = datatest_kl_NW[datatest_kl_NW.Image != 128]
target_kl=Dataset_Meta_kl.loc[:,"Rezidiv"].values
target_kl=target_kl.astype(int)
Dataset_Meta_kl_new=Dataset_Meta_kl.loc[:,["Nummer","pT","pN","pM"]]
Dataset_Meta_kl_new['Age_Diag'] = Dataset_Meta_kl['DiagDat'] - Dataset_Meta_kl['Geb']
Dataset_Meta_kl_new['Age_Diag'] = Dataset_Meta_kl_new["Age_Diag"] / timedelta(days=365)
# target_kl=[1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,0,1,1,0,0,1,0,0,0,0,1,1,0,1,0,1,1,0,0,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0,1,1,0,1,1,0,0,0,0,0,0,1,1,0,0,1,0,1]
Dataset_Meta_kl_new=Dataset_Meta_kl_new.rename(columns={"Nummer": "Image"})
# CLINICAL DATA included
datatest_kl_W=pd.merge(datatest_kl_W, Dataset_Meta_kl_new, on="Image")  #all wavelet
datatest_kl_NW=pd.merge(datatest_kl_NW, Dataset_Meta_kl_new, on="Image")  #no wavelet
datatest_kl_W=datatest_kl_W.drop('Image',axis=1)
datatest_kl_NW=datatest_kl_NW.drop('Image',axis=1)

## TRAIN AND TEST SET

#MERGING the 2 dataset
frame_W = [datatest_kl_W,datatest_RG_W]
datatest_full_W = pd.concat(frame_W)
target_full = np.concatenate((target_kl, target_RG), axis=None)
# Split FULL dataset into training set and test set
X_train_W, X_test1_W, y_train, y_test = train_test_split(datatest_full_W, target_full, test_size=0.3,random_state=100) # 70% training and 30% test

trainset_W=X_train_W
trainset_NW=X_train_W.iloc[:,0:107]
traintarget=y_train
testset_W=X_test1_W
testset_NW=X_test1_W.iloc[:,0:107]
testtarget=y_test

## OLD
# trainset_W=datatest_RG_W
# trainset_NW=datatest_RG_NW
# traintarget=target_RG
# testset_W=datatest_kl_W
# testset_NW=datatest_kl_NW
# testtarget=target_kl

## FEATURE SELECTION AND PARAMETER OPTIMISATION
#Get splits for trainset
split_W = defineSplits(trainset_W,traintarget,random_state=0)
split_NW = defineSplits(trainset_NW,traintarget,random_state=0)
#RUN the 3 classifier to get the best parameters
#NON WAVELET
svc_result_NW = optimise_SVC_featsel(trainset_NW,traintarget,cut=0.9,cv=split_NW)
logres_result_NW = optimise_logres_featsel(trainset_NW,traintarget,cut=0.9,metric='roc_auc',cv=split_NW)
rf_result_NW = optimise_rf_featsel(trainset_NW,traintarget,cut=0.9,cv=split_NW)
#WAVELET
svc_result_W = optimise_SVC_featsel(trainset_W,traintarget,cut=0.9,cv=split_W)
logres_result_W = optimise_logres_featsel(trainset_W,traintarget,cut=0.9,metric='roc_auc',cv=split_W)
rf_result_W = optimise_rf_featsel(trainset_W,traintarget,cut=0.9,cv=split_W)
#FIT THE MODEL WITH THE BEST PARAMS
#SVC
svc_result_NW.best_estimator_.fit(testset_NW,testtarget)
svc_result_W.best_estimator_.fit(testset_W,testtarget)
#LOGRES
logres_result_NW.best_estimator_.fit(testset_NW,testtarget)
logres_result_W.best_estimator_.fit(testset_W,testtarget)
#RANDOM FOREST
rf_result_NW.best_estimator_.fit(testset_NW,testtarget)
rf_result_W.best_estimator_.fit(testset_W,testtarget)
#PREDICTION:
#SVC
y_pred_svc_nw=svc_result_NW.best_estimator_.predict(testset_NW)
y_pred_svc_w=svc_result_W.best_estimator_.predict(testset_W)
#LOGRES
y_pred_logres_nw=logres_result_NW.best_estimator_.predict(testset_NW)
y_pred_logres_w=logres_result_W.best_estimator_.predict(testset_W)
#RANDOM FOREST
y_pred_rf_nw=rf_result_NW.best_estimator_.predict(testset_NW)
y_pred_rf_w=rf_result_W.best_estimator_.predict(testset_W)

##PLOT
#Confusion matrix
#SVC
plot_confusion_matrix(svc_result_NW, testset_NW, testtarget,cmap=plt.cm.Blues)  
plt.savefig('./ML_model_opt_plot/CF_SVC_Non_Wavelet.png',dpi=300, bbox_inches = "tight")
plot_confusion_matrix(svc_result_W, testset_W, testtarget,cmap=plt.cm.Blues)  
plt.savefig('./ML_model_opt_plot/CF_SVC_Wavelet.png',dpi=300, bbox_inches = "tight")
#LOGRES
plot_confusion_matrix(logres_result_NW, testset_NW, testtarget,cmap=plt.cm.Blues)  
plt.savefig('./ML_model_opt_plot/CF_LOGRES_Non_Wavelet.png',dpi=300, bbox_inches = "tight")
plot_confusion_matrix(logres_result_W, testset_W, testtarget,cmap=plt.cm.Blues)  
plt.savefig('./ML_model_opt_plot/CF_LOGRES_Wavelet.png',dpi=300, bbox_inches = "tight")
#RANDOM FOREST
plot_confusion_matrix(rf_result_NW, testset_NW, testtarget,cmap=plt.cm.Blues)  
plt.savefig('./ML_model_opt_plot/CF_RF_Non_Wavelet.png',dpi=300, bbox_inches = "tight")
plot_confusion_matrix(rf_result_W, testset_W, testtarget,cmap=plt.cm.Blues)  
plt.savefig('./ML_model_opt_plot/CF_RF_Wavelet.png',dpi=300, bbox_inches = "tight")

#ROC curve
def plot_something(model, trainset, testset, filename, ax=None, **kwargs):
    ax = ax
    y_train_pred_model = model.predict_proba(trainset)[::,1]    
    y_test_pred_model = model.predict_proba(testset)[::,1]
    train_fpr, train_tpr, tr_thresholds = roc_curve(traintarget, y_train_pred_model)
    test_fpr, test_tpr, te_thresholds = roc_curve(testtarget, y_test_pred_model)
    
    ax.plot(train_fpr, train_tpr, label=" AUC TRAIN ="+str(auc(train_fpr, train_tpr)))
    ax.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
    ax.legend()
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(filename)
    ax.grid(color='black', linestyle='-', linewidth=0.5)
    return ax

#SVC
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))
plot_something(svc_result_NW, trainset_NW, testset_NW, 'AUC(ROC curve)_SVC_NonWavelet', ax1)
plot_something(svc_result_W, trainset_W, testset_W, 'AUC(ROC curve)_SVC_Wavelet', ax2)
plt.savefig('./ML_model_opt_plot/AUC_SVC.png',dpi=300, bbox_inches = "tight")
#LOGRES
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))
plot_something(logres_result_NW, trainset_NW, testset_NW, 'AUC(ROC curve)_LOGRES_NonWavelet', ax1)
plot_something(logres_result_W, trainset_W, testset_W, 'AUC(ROC curve)_LOGRES_Wavelet', ax2)
plt.savefig('./ML_model_opt_plot/AUC_LOGRES.png',dpi=300, bbox_inches = "tight")
#RANDOM FOREST
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))
plot_something(rf_result_NW, trainset_NW, testset_NW, 'AUC(ROC curve)_RF_NonWavelet', ax1)
plot_something(rf_result_W, trainset_W, testset_W, 'AUC(ROC curve)_RF_Wavelet', ax2)
plt.savefig('./ML_model_opt_plot/AUC_RF.png',dpi=300, bbox_inches = "tight")
