import os 
import sys 
import dill

import numpy as np
import pandas as pd

from src.exception import CustomException
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True) 

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        best_estimators = {}

        for model_name, model in models.items():
            para = params.get(model_name, {})

            if para:
                rs = RandomizedSearchCV(model, para, n_iter=30, cv=3, n_jobs=-1, random_state=42)
                rs.fit(X_train, y_train)
                model.set_params(**rs.best_params_)

            model.fit(X_train, y_train)

            y_test_proba = model.predict_proba(X_test)[:, 1]
            test_model_score = roc_auc_score(y_test, y_test_proba)

            report[model_name] = test_model_score
            best_estimators[model_name] = model

        return report, best_estimators

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path ):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)