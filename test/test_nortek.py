import os
import unittest

import numpy as np
import pandas as pd

from advmdata.nortek import EzqADVMData


class TestEzqADVMData(unittest.TestCase):
    """Test the public methods of the EzqADVMData class"""

    def test_read_ezq_data(self):
        """Test the read_ezq_data method and compare it to the expected results"""

        data_set_path = os.path.join('data', 'nortek', 'EZQ', 'EZQ')
        ezq_advm_data = EzqADVMData.read_ezq_data(data_set_path)
        read_results_df = ezq_advm_data.get_data()

        results_path = data_set_path + '_results.txt'
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
        read_results_origin = ezq_advm_data.get_origin()

        expected_origin_path = data_set_path + '_expected_origin.txt'
        expected_origin = pd.read_table(expected_origin_path, index_col=0)

        self.assertTrue(np.all(read_results_origin == expected_origin))


if __name__ == '__main__':
    unittest.main()
