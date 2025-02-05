# I want to predict for the new data I will basically write over here as predict

import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object 


class PredictPipeline:
    

    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'

            model = load_object(file_path = model_path)
            preprocessor = load_object(preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)
    
class CustomData:
    # custom data
    # 12:06
    # class will be responsible in mapping all
    # 12:08
    # the inputs that we are giving in the
    # 12:10
    # HTML to the back end with this
    # 12:11
    # particular values

    def __init__(self,
                 gender: str,
                 race_ethnicity: int,
                 parental_level_of_education,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        
        self.gender = gender
        self.race_ethnicity  = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):

        # this function will do is that it
        # 14:27
        # will just return all my input in the
        # 14:29
        # form of a data frame because we train
        # 14:31
        # our model in the form of a data frame

        try:
            custom_data_input_dict = {
                "gender":[self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
        