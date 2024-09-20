import os
import sys

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
import numpy as np 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import warnings
from src.utils import save_path,evaluate_models


@dataclass

class Modeltrainerconfig:
    train_model_file_path = os.path.join('artifact','model.pkl')
class Modeltrainer:
    def __init__(self):
        self.model_trainer_config = Modeltrainerconfig()
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("spliting training  and test data")
            x_train,y_train,x_test,y_test = (
                train_arr[:, :-1], train_arr[:, -1],
                test_arr[:, :-1], test_arr[:, -1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "SVR":SVR(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report:dict = evaluate_models(x_train,y_train,x_test,y_test,models)

            max_r2_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(max_r2_score)
            ]

            best_model = models[best_model_name]

            if max_r2_score <0.6:
                raise CustomException("no best model found")
            logging.info(f"Best model found on the training and testing phase is: {best_model_name}")

            save_path(
                self.model_trainer_config.train_model_file_path,
                best_model
            )
            predicted = best_model.predict(x_test)
            r2_sc = r2_score(y_test,predicted)

            return r2_sc    


        except Exception as e:
            raise CustomException(e,sys)
