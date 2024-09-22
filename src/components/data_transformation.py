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
class Datatransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')

class Datatransformation:
    def __init__(self):
        self.datatransformation_config = Datatransformationconfig()
    
    def get_data_transformer_obj(self):
        try:
            # Define the numerical and categorical features
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"Numerical columns: {numerical_features}")

            # Column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_features),
                    ("cat", cat_pipeline, categorical_features)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read the training and testing datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading training and testing datasets completed")
            
            # Preprocessing objects
            preprocessor_obj = self.get_data_transformer_obj()

            target_column = 'math_score'
            numerical_features = ['reading_score', 'writing_score']

            # Separating input features and target features for training and testing data
            input_train_data = train_df.drop(columns=[target_column], axis=1)
            target_train_data = train_df[target_column]

            input_test_data = test_df.drop(columns=[target_column], axis=1)
            target_test_data = test_df[target_column]

            logging.info("Applying preprocessing on training and test data")

            # Apply preprocessing to training and testing data
            input_feature_train = preprocessor_obj.fit_transform(input_train_data)
            input_feature_test = preprocessor_obj.transform(input_test_data)

            # Combine the processed input features with the target variable
            train_arr = np.c_[input_feature_train, np.array(target_train_data)]
            test_arr = np.c_[input_feature_test, np.array(target_test_data)]

            logging.info("Saving preprocessing object")

            # Save the preprocessor object
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
            raise CustomException(e, sys)
