import sys
import os
from dataclasses import dataclass




import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # to build pipeline
from sklearn.impute import SimpleImputer # to handle missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler # to handle categorical,numerical transformations

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig: #this class creates paths for data transformation
    preprocessor_obj_file_path=os.path.join('artifact',"preprocessor.pkl") # this path converts objects to pickle files

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig() # moves preprocessor_obj_file_path to data_transformation_config

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation base on the type of data - numerical, categorical
        '''
        try:
            numerical_columns=["writing score","reading score"]
            categorical_columns=[
                "gender", "race/ethnicity", 
                "parental level of education",
                "lunch",
                "test preparation course",
            ]


            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")), # handles missing data
                    ("scaler",StandardScaler(with_mean=False)) # ensures numerical stability
                ]

            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]


            )
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor=ColumnTransformer( # combines numerical,categorical pipelines & transforms accordingly
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)

                ]

            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def inititate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train,test data completed")
            logging.info("Obtaining the preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()
            target_column_name="math score"
            numerical_columns=["writing score","reading score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying the preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[ #np.c_[]:combines input features & target labels(stacking arrays column-wise)
                input_feature_train_arr, np.array(target_feature_train_df) 
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object( # saves the pickle file from preprocessor_obj_file_path

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        