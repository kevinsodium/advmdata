import os
import unittest

import numpy as np
import pandas as pd

from advmdata.nortek import EZQADVMData


class TestEZQADVMDataInit(unittest.TestCase):
    """Test the successful initialization of the EZQADVMData class"""

    def setUp(self):
        """Initialize an instance of EZQADVMData to test"""

        self.data_set_path = os.path.join('data', 'nortek', 'EZQ', 'EZQ')
        self.ezq_advm_data = EZQADVMData.read_ezq_data(self.data_set_path, cell_size=0.4)

    def test_read_ezq_data(self):
        """Test the read_ezq_data method and compare the relevant acoustic data to the expected results"""

        read_results_df = self.ezq_advm_data.get_data()

        results_path = self.data_set_path + '_results.txt'
        expected_results_df = pd.read_table(results_path, index_col=0, parse_dates=True)

        # test all times are being read correctly
        self.assertTrue(np.all(read_results_df.index == expected_results_df.index))

        # test all columns are being read correctly
        self.assertTrue(np.all(read_results_df.keys() == expected_results_df.keys()))

        # test all values read are equal to expected results
        # this is messy because np.nan == np.nan is not True
        number_of_columns = read_results_df.shape[1]
        column_results_series = pd.Series(data=np.tile(False, (number_of_columns, )), index=read_results_df.keys())
        for column in read_results_df.keys():
            column_results_series[column] = np.all(read_results_df[column].dropna()
                                                   == expected_results_df[column].dropna())
        self.assertTrue(np.all(column_results_series))

        # test the origin is created correctly
        read_results_origin = self.ezq_advm_data.get_origin()

        expected_origin_path = self.data_set_path + '_expected_origin.txt'
        expected_origin = pd.read_table(expected_origin_path, index_col=0)

        self.assertTrue(np.all(read_results_origin == expected_origin))


if __name__ == '__main__':
    unittest.main()
