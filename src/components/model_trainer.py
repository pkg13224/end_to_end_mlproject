# basically all the training code over here how many different kinds of model I want to use I will probably also call
# over here the confusion metrics if I'm probably solving the classification problem if I'm solving a regression
# problem I may see R2 r squared adjusted r squared value all those things will basically happen over here right and
# from here also uh see next one more step I can basically write model Pusher also okay but I will not still make 

# Basic Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import sys
# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException
import os
from dataclasses import dataclass
from src.utils import evaluate_models
# every component we really
# need to create a config class
# and there I will be giving some path like of
# the pickle file whatever input is basically required

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1])

            models = {
                        "Linear Regression": LinearRegression(),
                        "Lasso": Lasso(),
                        "Ridge": Ridge(),
                        "K-Neighbors Regressor": KNeighborsRegressor(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "Random Forest Regressor": RandomForestRegressor(),
                        "XGBRegressor": XGBRegressor(), 
                        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                        "AdaBoost Regressor": AdaBoostRegressor()
                    }
            
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test= X_test, y_test= y_test, models = models)
            
            # Get best r2_score
            best_model_score = max(sorted(model_report.values()))

            # Get the model name that has heigest r2_score
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name] # this is best model obj coming from models dict.

            if best_model_score<0.6:
                raise CustomException("No best Model found.")
        
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            ) 

            predicted = best_model.predict(X_test)
            r2= r2_score(y_test, predicted)
            return r2

        except Exception as e:
            raise CustomException(e, sys)
        