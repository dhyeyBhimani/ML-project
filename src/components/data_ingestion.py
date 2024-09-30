import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import Datatransformation
from src.components.data_transformation import Datatransformationconfig

from src.components.data_training import ModelTrainer
from src.components.data_training import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path:str  = os.path.join('artifact','train.csv')
    test_data_path:str  = os.path.join('artifact','test.csv')
    raw_data_path:str  = os.path.join('artifact','raw.csv')
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    def initiated_data_ingestion(self):
        logging.info("Enter the data Ingestion method or component")
        try:
            df = pd.read_csv('Notebook/Data/stud.csv')
            logging.info("read dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("train test split initiated")
            train_data,test_data = train_test_split(df,test_size = 0.20,random_state = 42)

            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("ingestion of the data is complete")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except CustomException as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiated_data_ingestion()

    data_transfomation = Datatransformation()
    train_arr,test_arr,_= data_transfomation.initiate_data_transformation(train_data,test_data)

    data_training = ModelTrainer()
    print(data_training.initiate_model_trainer(train_arr,test_arr))
