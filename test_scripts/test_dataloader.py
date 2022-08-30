import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

from scripts.dataloader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.dir_name = "../data"

    def test_read_csv(self):
        """Test the readcsv method"""
        filename = "AdSmartABdata.csv"
        dataloader = DataLoader()
        pd = dataloader.read_data(self.dir_name, filename)
        col1 = pd.columns[0]
        self.assertEqual(col1, "auction_id")

if __name__ == '__main__':
    unittest.main()