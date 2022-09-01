import numpy as np 


class Utils:
    """A class for helper functions
    """

    def get_missing_info(self, df):
        """A method for missing information
        Attributes:
            df: Pandas dataframe

        Returns:
            count: The number of missing data
            percent: The % of missing data
        """
        totalCells = np.product(df.shape)
        missingCount = df.isnull().sum()
        totalMissing = missingCount.sum()
        missing_percent = round((totalMissing/totalCells), 2) * 100
        return missingCount, missing_percent

    def get_unique_values(self, df):
        """A method to get unique values of a column
        Attributes:
            df: Dataframe

        Returns:
            unique_values: Unique elements
        """
        return df.value_counts()

