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
    expected_config_dict = None

    def _test_cell_range(self):
        """Tests the calculation of the cell range"""
        calculated_cell_range = self.advm_data.get_cell_range().mean()

        # create a Series with the expected cell ranges
        number_of_cells = self.expected_config_dict['Number of Cells']
        blanking_distance = self.expected_config_dict['Blanking Distance']
        cell_size = self.expected_config_dict['Cell Size']

        cell_numbers = np.arange(1, number_of_cells + 1)
        expected_cell_range_index = ['R{:03}'.format(cell) for cell in cell_numbers]
        expected_cell_range_data = blanking_distance + cell_numbers * cell_size
        expected_cell_range = pd.Series(data=expected_cell_range_data, index=expected_cell_range_index)

        pd.testing.assert_series_equal(calculated_cell_range, expected_cell_range)

    def _test_config_parameters(self):
        """Test the configuration parameters"""

        result_config_dict = self.advm_data.get_configuration_parameters().get_dict()
        self.assertEqual(result_config_dict, self.expected_config_dict)

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

        self.expected_config_dict = {'Beam Orientation': 'Horizontal',
                                     'Blanking Distance': 0.2,
                                     'Cell Size': 0.2,
                                     'Effective Transducer Diameter': 0.01395,
                                     'Frequency': 2000.0,
                                     'Instrument': 'AQD',
                                     'Number of Beams': 2,
                                     'Number of Cells': 50,
                                     'Slant Angle': 25.0}

    def test_aquadopp_cell_range(self):
        """Test the calculation of the cell range for the Aquadopp"""
        self._test_cell_range()

    def test_aquadopp_config_parameters(self):
        """Test the configuration parameters from reading the Aquadopp data"""
        self._test_config_parameters()

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
        self.expected_config_dict = {'Beam Orientation': 'Horizontal',
                                     'Blanking Distance': 0.2,
                                     'Cell Size': 0.4,
                                     'Effective Transducer Diameter': 0.01395,
                                     'Frequency': 1000.0,
                                     'Instrument': 'EZQ',
                                     'Number of Beams': 4,
                                     'Number of Cells': 64,
                                     'Slant Angle': 25.0}

        test_data_set = 'EZQ'
        self.data_set_path = os.path.join(ezq_data_path, test_data_set)
        self.advm_data = EZQADVMData.read_ezq_data(self.data_set_path, cell_size=self.expected_config_dict['Cell Size'])

    def test_ezq_cell_range(self):
        """Test the calculation of the cell range for the EZQ"""
        self._test_cell_range()

    def test_ezq_config_parameters(self):
        """Test the configuration parameters from reading the EZQ data"""
        self._test_config_parameters()

    def test_ezq_data(self):
        """Test the read_ezq_data method and compare the relevant acoustic data to the expected results"""

        self._test_data()

    def test_ezq_origin(self):
        """Test the creation of the data origin"""
        self._test_origin()


if __name__ == '__main__':
    unittest.main()
