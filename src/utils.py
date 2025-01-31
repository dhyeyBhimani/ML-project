import os
import sys
from src.exception import CustomException
import pickle
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import GridSearchCV

def save_path(path_name,obj):
    try:
        path = os.path.dirname(path_name)
        os.makedirs(path,exist_ok=True)
        with open(path_name, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        pass

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=5)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e, sys)
    
