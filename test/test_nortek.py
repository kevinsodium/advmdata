import os
import unittest

import numpy as np

from advmdata.nortek import AquadoppADVMData, EZQADVMData
from test.test_core import TestADVMDataInit

current_path = os.path.dirname(os.path.realpath(__file__))

# relative top level path of nortek data
nortek_data_path = os.path.join(current_path, 'data', 'nortek')

# Aquadopp test data path
aqd_data_path = os.path.join(nortek_data_path, 'Aquadopp')

# EZQ test data path
ezq_data_path = os.path.join(nortek_data_path, 'EZQ')


class TestNortekADVMDataInit(TestADVMDataInit):

    def _get_expected_cell_range_data(self):

        number_of_cells = self.expected_config_dict['Number of Cells']
        blanking_distance = self.expected_config_dict['Blanking Distance']
        cell_size = self.expected_config_dict['Cell Size']

        cell_numbers = np.arange(1, number_of_cells + 1)
        expected_cell_range_data = blanking_distance + cell_numbers * cell_size

        return expected_cell_range_data


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
                                     'Number of Cells': 10,
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
        """Test the creation of the Aquadopp data origin"""
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
                                     'Number of Cells': 10,
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
