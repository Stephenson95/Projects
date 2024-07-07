# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 23:23:00 2024

@author: Stephenson
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
import optuna

#Reproducibility
seed = 0

#Set import path
import_path = r"C:\Users\{}\Documents\GitHub\Projects\Insurance Cross Selling".format(os.getlogin())
train_data = pd.read_csv(import_path + r'\data\train.csv')
train_data.drop('id', axis = 'columns', inplace=True)

#Split labels and data
X = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]

#Initialise encoders
scaler = StandardScaler()
encoder = OneHotEncoder(sparse_output = False)

#Categorise Columns
cat_cols = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
num_cols = list(set(X.columns) - set(cat_cols))

#Initialise Transformer
transformer = ColumnTransformer([('cat_cols', encoder, cat_cols),
                                ('num_cols', scaler, num_cols)])

#%%
#Initial Fit

#Initialise model
model = xgb.XGBClassifier()

#Create pipeline
pipe = Pipeline([("preprocessing", transformer),
                ("model", model)])

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

#Fit and score
pipe.fit(X_train,y_train).score(X_test, y_test)

#%%
#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

#Transform
X_train = transformer.fit_transform(X_train)
X_test = transformer.transform(X_test)


def objective(trial):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)

    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(y_test, pred_labels)
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200, timeout=600)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


#%%
#CV result

#Set fold
kf = KFold(n_splits = 10, shuffle = True, random_state=seed)

test_scores = []


for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    X_train_fold, X_val_fold = X.iloc[train_index,:], X.iloc[val_index,:]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    # Prepare the model
    model = xgb.XGBClassifier()
    model.set_params(**trial.params)
    model.set_params(device='gpu', seed = seed)

    # Fit model on train fold and use validation for early stopping
    pipe = Pipeline([("preprocessing", transformer),
                    ("model", model)])
    pipe.fit(X_train_fold,y_train_fold)
    
    # Predict on test set
    y_pred_test = pipe.predict(X_val_fold)
    test_score = accuracy_score(y_val_fold, y_pred_test)
    test_scores.append(test_score)
    
    print(r"Fold {}: Accuracy {}".format(fold, test_score))

# Compute average score across all folds
average_score = np.mean(test_scores)

with open(import_path + r"\outputs\XGB_scores.txt", "w") as output:
    output.write("Scores: ")
    output.write(str(test_scores))
    output.write("\nAverage: ")
    output.write(str(average_score))
#%%
#Retrain and save model
model = xgb.XGBClassifier()
model.set_params(**trial.params)
model.set_params(device='gpu', seed = seed)

# Fit model on train fold and use validation for early stopping
pipe = Pipeline([("preprocessing", transformer),
                ("model", model)])

pipe.fit(X,y)
joblib.dump(pipe, import_path + r'\outputs\XGB.pkl')
#%%
#Submission

#Load model
final_pipeline = joblib.load(import_path + r'\outputs\XGB.pkl')

#Load test data
test_data = pd.read_csv(import_path + r'\data\test.csv')
output_id = test_data['id']
test_data.drop('id', axis = 'columns', inplace=True)

# Use the loaded pipeline model to make predictions
final_pred = final_pipeline.predict(test_data)

#Output
submissionfile = pd.concat([pd.Series(output_id), pd.Series(final_pred)], axis = 1)
submissionfile.rename(columns = {0:'Response'}, inplace=True)

submissionfile.to_csv(import_path + r'\submission\submission_xgb.csv', index=False)

