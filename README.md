# ML_opt_model
Machine Learning method for predicting the relapse status of a patients based on a dataset of Radiomics Features extracted with Pyradiomics.
The script take these inputs plus the metadata of the patients containing the relapse class status, and other clinical informations
that can be used to add more features to the model.
## Model description
The ML model consist of a pipeline with multiple steps:

-Correlation: dropping features too correlated between each others

-Scaling: features scaling

-Best k features: Selecting the k best features in the set

-ML classifier: prediction of the relapse using SVM, Logistic Regression and Random Forest
## Plot
Confusion Matrix and ROC AUC plots are then generated to evaluate the performance of the model.
