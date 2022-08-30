import pandas as pd 
import os

class DataLoader:
    """A class for data reading
    """

    def read_data(self, dir_name, file_name):
        """A method for reading csv files
        Attrs:
            dir_name: Folder of the file
            file_name: The name of the file
             
        return:
            pandas dataframe
        """
        os.chdir(dir_name)
        df = pd.read_csv(file_name)
        return df