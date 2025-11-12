#it will have all the common utility functions which can be used in entire project and the import statements
import os
import sys
import pandas as pd
import dill
import numpy as np
from src.exception import Customexception
from sklearn.metrics import r2_score, mean_squared_error

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb')as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise Customexception(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            #train the model
            model.fit(X_train,y_train)

            #predicting the test data
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            #r2 score
            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]=test_model_score
        return report
    except Exception as e:
        raise Customexception(e,sys)