#feature cleaning and data preprocessing
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer #used to create piplines we can use this column transformer library
from sklearn.impute import SimpleImputer #used for handling missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import Customexception
from src.logger import logging

from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.Data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns=['reading score','writing score']
            categorical_columns=['gender','race/ethnicity','parental level of education','lunch','test preparation course']
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info("numerical pipeline:{numerical_columns}")
            logging.info("categorical pipeline:{categorical_columns}")

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
                ])
            return preprocessor
        except Exception as e:
            raise Customexception(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("read train and test data completed")
            logging.info("obtaining preprocessor object")

            preprocessor_obj=self.get_data_transformer_obj()

            target_col="math score"
            numerical_columns=['reading score','writing score']

            input_feature_train_df=train_df.drop(columns=[target_col],axis=1)
            target_feature_train_df=train_df[target_col]

            input_feature_test_df=test_df.drop(columns=[target_col],axis=1)
            target_feature_test_df=test_df[target_col]
            logging.info(
                f"applying preprocessor obj on training dataframe and testing dataframe."
            )
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info(f"saved preprocessor object.")

            save_object(
                file_path=self.Data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return(
                train_arr,
                test_arr,
                self.Data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise Customexception(e,sys)