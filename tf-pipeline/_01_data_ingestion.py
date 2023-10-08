import numpy as np
import pandas as pd
import os
import kaggle

def load_data():
 
    # Kaggle identifiers are defined in Github as secrets

    # Download the dataset by specifying the dataset name
    dataset_name = 'msambare/fer2013'
    kaggle.api.dataset_download_files(dataset_name, path='tf-pipeline', unzip=True)

