import os
import sys
from src.exception import CustomException
import pickle

def save_path(path_name,obj):
    try:
        path = os.path.dirname(path_name)
        os.makedirs(path,exist_ok=True)
        with open(path_name, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        pass
