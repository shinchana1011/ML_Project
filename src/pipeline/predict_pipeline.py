#A prediction pipeline is the end-to-end process that takes new, unseen input data and produces a model prediction — using all the same steps that were applied during training.

import pickle
import sys
import pandas as pd
from src.exception import Customexception
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor_path=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor_path.transform(features)
            pred=model.predict(data_scaled)
            return pred
        except Exception as e:
            raise Customexception(e,sys)
    
class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score]
            }

            df = pd.DataFrame(custom_data_input_dict)
            print("DataFrame being sent to preprocessor:\n", df.head())  # Debug
            return df

        except Exception as e:
            raise Customexception(e, sys)
