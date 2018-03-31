import os
import unittest

import numpy as np
import pandas as pd

from advmdata.nortek import AquadoppADVMData, EZQADVMData

# relative top level path of nortek data
nortek_data_path = os.path.join('data', 'nortek')

# Aquadopp test data path
aqd_data_path = os.path.join(nortek_data_path, 'Aquadopp')

# EZQ test data path
ezq_data_path = os.path.join(nortek_data_path, 'EZQ')


class TestNortekADVMDataInit(unittest.TestCase):

    advm_data = None
    data_set_path = None

    def _test_data(self):
        """Test the accuracy of the data set read by the read method"""

        read_results_df = self.advm_data.get_data()

        results_path = self.data_set_path + '_results.txt'
        expected_results_df = pd.read_table(results_path, index_col='DateTime', parse_dates=True)

        pd.testing.assert_frame_equal(read_results_df, expected_results_df)

    def _test_origin(self):

        # test the origin is created correctly
        read_results_origin = self.advm_data.get_origin()

        expected_origin_path = self.data_set_path + '_expected_origin.txt'
        expected_origin = pd.read_table(expected_origin_path, index_col=0)

        pd.testing.assert_frame_equal(read_results_origin, expected_origin)


class TestAquadoppADVMDataInit(TestNortekADVMDataInit):
    """Test the successful initialization of the AdquadoppADVMData class"""

    def setUp(self):
        """Initialize an instance of the AquadoppADVMData class"""
        test_data_set = 'AQD.prf'
        self.data_set_path = os.path.join(aqd_data_path, test_data_set)
        self.advm_data = AquadoppADVMData.read_aquadopp_data(self.data_set_path)

    def test_aquadopp_data(self):
        """Test the accuracy of the reading of the Aquadopp data set"""
        self._test_data()

    def test_aquadopp_origin(self):
        """Test the creation of the data origin"""
        self._test_origin()


class TestEZQADVMDataInit(TestNortekADVMDataInit):
    """Test the successful initialization of the EZQADVMData class"""

    def setUp(self):
        """Initialize an instance of EZQADVMData to test"""
        test_data_set = 'EZQ'
        self.data_set_path = os.path.join(ezq_data_path, test_data_set)
        self.advm_data = EZQADVMData.read_ezq_data(self.data_set_path, cell_size=0.4)

    def test_ezq_data(self):
        """Test the read_ezq_data method and compare the relevant acoustic data to the expected results"""

        self._test_data()

    def test_ezq_origin(self):
        """Test the creation of the data origin"""
        self._test_origin()


if __name__ == '__main__':
    unittest.main()
