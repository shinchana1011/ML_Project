#model trainer.py
import os
import sys
from dataclasses import dataclass
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ML Models
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

# Custom imports
from src.exception import Customexception
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    # path to save the trained pickle model
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("split training and test input data")

            # ✅ Correct array slicing
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],   # all rows, all columns except last column
                train_array[:, -1],    # last column is target
                test_array[:, :-1],
                test_array[:, -1]
            )

            # ✅ All candidate ML models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # ✅ Evaluate all models (your logic preserved)
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)

            logging.info(f"Model performance report: {model_report}")

            # ✅ Your intention: select best model based on score
            #    model_report structure is {model_name: score}
            best_model_score = max(model_report.values())  # highest R2
            best_model_name = max(model_report, key=model_report.get)  # name with highest R2
            best_model = models[best_model_name]  # fitted instance (evaluate_model fit in-place)

            logging.info(f"Best Model: {best_model_name} with score: {best_model_score}")

            # ✅ Your condition: if best model score < 0.6 throw exception
            if best_model_score < 0.6:
                raise Customexception("No best model found (r2 score < 0.6)")

            # ✅ Save the best model as pickle file
            # We use this pickle file to save model transformation
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # ✅ Calculate R2 score on test set
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise Customexception(e, sys)
