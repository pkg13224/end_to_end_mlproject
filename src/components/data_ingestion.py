# This file has code for data reading and divide data in train and test.
import os
import sys
from src.exception import CustomException 
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split 
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

# Inside a class to define 
# class variable we need to use init
# funtion. But using this dataclass
# variable we can directly define class
# variable.
@dataclass 
class DataIngestionConfig: 
    # Any input requried to data ingestion component, will be given through
    # this DataIngestionConfig class.
    # like train path, test path, raw data path
    
    # These path are input to data ingestion class and output of data ingestion class will be saved on these path.
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        
        logging.info('Entered the data ingestion method or component')
        try:
            """
            Here we will read the data from source like database or some other source
            and do train test split and store on the path received from DataIngestionConfig class.
            We could have also done data reading in utils.py and directly call that funtion in this funtion.
            """
            df = pd.read_csv('notebooks\data\stud.csv')
            logging.info('Read the dataset as dataframe.')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header  = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info('Ingestion of the data is completed.')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )

        except Exception as e:
            logging.info("")
            raise CustomException(e, sys)


if __name__ =="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    modeltrainer = ModelTrainer()
    r2 = modeltrainer.initiate_model_trainer(train_array=train_arr, test_array= test_arr)
    logging.info(f'Best model r2 socre: {r2}')