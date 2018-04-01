import os
import unittest

import numpy as np

from advmdata.sontek import ArgonautADVMData, SL3GADVMData
from test.test_core import TestADVMDataInit

current_path = os.path.dirname(os.path.realpath(__file__))

# relative top level path of nortek data
sontek_data_path = os.path.join(current_path, 'data', 'sontek')

# Aquadopp test data path
arg_data_path = os.path.join(sontek_data_path, 'Argonaut')

# EZQ test data path
sl3g_data_path = os.path.join(sontek_data_path, 'SL3G')


class TestArgonautADVMDataInit(TestADVMDataInit):
    """Test the successful initialization of the ArgonautADVMData class"""

    def setUp(self):
        """Initialize instance of ArgonautADVMData class"""

        test_data_set = 'ARG1'
        self.data_set_path = os.path.join(arg_data_path, test_data_set)
        self.advm_data = ArgonautADVMData.read_argonaut_data(self.data_set_path)

        self.expected_config_dict = {'Beam Orientation': 'Horizontal',
                                     'Blanking Distance': 1.,
                                     'Cell Size': 1.75,
                                     'Effective Transducer Diameter': 0.03,
                                     'Frequency': 1500.0,
                                     'Instrument': 'SL',
                                     'Number of Beams': 2,
                                     'Number of Cells': 10,
                                     'Slant Angle': 25.0}

    def _get_expected_cell_range_data(self):
        """Calculate the cell range expected for the Argonaut instrument"""

        number_of_cells = self.expected_config_dict['Number of Cells']
        blanking_distance = self.expected_config_dict['Blanking Distance']
        cell_size = self.expected_config_dict['Cell Size']

        cell_numbers = np.arange(0, number_of_cells)
        expected_cell_range_data = blanking_distance + 0.5 * cell_size + (cell_numbers * cell_size)

        return expected_cell_range_data

    def test_argonaut_cell_range(self):
        """Test the calculate of the cell range for the Argonaut"""
        self._test_cell_range()

    def test_argonaut_config_parameters(self):
        """Test the configuration parameters from reading the Argonaut data"""
        self._test_config_parameters()

    def test_argonaut_data(self):
        """Test the accuracy of the reading of the Argonaut data set"""
        self._test_data()

    def test_origin(self):
        """Test the creation of the Argonaut data origin"""
        self._test_origin()


if __name__ == '__main__':
    unittest.main()
