#it will have all the common utility functions which can be used in entire project and the import statements
import os
import sys
import pandas as pd
import dill
import numpy as np
from src.exception import Customexception

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb')as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise Customexception(e,sys)

