import os
import sys
import numpy as np
import pandas as pd
import pickle
import dill
from sklearn.metrics import r2_score



from src.exception import CustomException
def save_object(file_path, obj): #the path to save to, the object to be saved
    try:
        dir_path = os.path.dirname(file_path) #gets a folder from the file_path

        os.makedirs(dir_path, exist_ok=True) #creates a folder off the file_path

        with open(file_path, "wb") as file_obj: # opens file with 'write binary' mode
            dill.dump(obj, file_obj) #converts file to storable format using 'dill.dump'(serialized)

    except Exception as e:
        raise CustomException(e, sys) 