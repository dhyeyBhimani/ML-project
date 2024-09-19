import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_path

@dataclass
class Datatransformationconfig():
    preprocessor_obj_file_path = os.path.join('artifact','preprocessor.pkl')

class Datatransformation:
    def __init__(self):
        self.datatransformation_config = Datatransformationconfig()
    def get_data_transformer_obj(self):
        try:
            numrical_feature = ['reading_score', 'writing_score']
            cat_feature = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehot",OneHotEncoder()),
                ("sclar",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {cat_feature}")
            logging.info(f"Numerical columns: {numrical_feature}")

            preprocessor = ColumnTransformer(
                [
                ("num", num_pipeline, numrical_feature),
                ("cat", cat_pipeline, cat_feature)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("reading training and testing is completed")
            logging.info("obtaining preprocessing objects")

            preprocessor_obj = self.get_data_transformer_obj()
            target_column = ['math_score']
            numrical_feature = ['reading_score', 'writing_score']

            input_train_data = train_df.drop(columns=['math_score'],axis = 1)
            target_feature__train_name= train_df[target_column]

            input_test_data = test_df.drop(columns=['math_score'],axis = 1)
            target_feature__test_name= test_df[target_column]

            logging.info("applying preprocessing onj on train and test data")

            input_feature_train = preprocessor_obj.fit_transform(input_train_data)
            input_test_data = preprocessor_obj.transform(input_test_data)

            train_arr = np.c_[input_train_data,np.array(target_feature__train_name)]
            test_arr = np.c_[input_test_data,np.array(target_feature__test_name)]

            logging.info("save preprocessing object")

            save_path(
                self.datatransformation_config.preprocessor_obj_file_path,
                preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.datatransformation_config.preprocessor_obj_file_path
            )




        except Exception as e:
            raise CustomException(e,sys)

        
