import os
import unittest

import numpy as np

from advmdata.sontek import ArgonautADVMData, SL3GADVMData

from test.test_core import TestADVMDataInit, TestADVMDataAddData

current_path = os.path.dirname(os.path.realpath(__file__))

# relative top level path of nortek data
sontek_data_path = os.path.join(current_path, 'data', 'sontek')

# Aquadopp test data path
arg_data_path = os.path.join(sontek_data_path, 'Argonaut')

# EZQ test data path
sl3g_data_path = os.path.join(sontek_data_path, 'SL3G')


class TestArgonautADVMDataInit(TestADVMDataInit):
    """Test the initialization of the ArgonautADVMData class"""

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


class TestSL3GADVMDataInit(TestADVMDataInit):
    """Test the initialization of the SL3GADVMData class"""

    def setUp(self):
        """Initialize instance of SL3GADVMData class"""

        test_data_set = 'SL3G1.mat'
        self.data_set_path = os.path.join(sl3g_data_path, test_data_set)
        self.advm_data = SL3GADVMData.read_sl3g_mat(self.data_set_path)

        self.expected_config_dict = {'Beam Orientation': 'Horizontal',
                                     'Blanking Distance': 0.2,
                                     'Cell Size': 0.2,
                                     'Effective Transducer Diameter': None,
                                     'Frequency': 1500.0,
                                     'Instrument': 'SL3G',
                                     'Number of Beams': 2,
                                     'Number of Cells': 68,
                                     'Slant Angle': 25.0}

    def _get_expected_cell_range_data(self):
        """Calculate the cell range expected for the Argonaut instrument"""

        number_of_cells = self.expected_config_dict['Number of Cells']
        blanking_distance = self.expected_config_dict['Blanking Distance']
        cell_size = self.expected_config_dict['Cell Size']

        cell_numbers = np.arange(0, number_of_cells)
        expected_cell_range_data = blanking_distance + 1.5* cell_size + (cell_numbers * cell_size)

        return expected_cell_range_data

    def test_sl3g_cell_range(self):
        """Test the calculate the cell range for the SL3G"""
        self._test_cell_range()

    def test_sl3g_config_parameters(self):
        """Test the configuration parameters from reading the SL3G data"""
        self._test_config_parameters()

    def test_sl3g_data(self):
        """Test the accuracy of the reading of the SL3G data set"""
        # read_results_df = self.advm_data.get_data()

        # results_path = self.data_set_path + '_results.txt'
        # read_results_df.to_csv(results_path, sep='\t')
        self._test_data()

    def test_sl3g_origin(self):
        """Test the creation of the SL3G data origin"""
        # read_results_origin = self.advm_data.get_origin()

        # expected_origin_path = self.data_set_path + '_expected_origin.txt'
        # read_results_origin.to_csv(expected_origin_path, sep='\t')
        self._test_origin()


class TestArgonautADVMDataAddData(TestADVMDataAddData):
    """Test adding ADVMData instances"""

    read_data_method = ArgonautADVMData.read_argonaut_data

    def setUp(self):
        """Define data set paths"""

        # data sets 1 and 2 are compatible and complete (VEL, DAT, SNR, CTL)
        data_set_1 = 'ARG1'
        self.data_set_1_path = os.path.join(arg_data_path, data_set_1)
        data_set_2 = 'ARG2'
        self.data_set_2_path = os.path.join(arg_data_path, data_set_2)

        # data sets 3 and 4 are compatible and incomplete (DAT, VEL only)
        data_set_3 = 'ARG3'
        self.data_set_3_path = os.path.join(arg_data_path, data_set_3)
        data_set_4 = 'ARG4'
        self.data_set_4_path = os.path.join(arg_data_path, data_set_4)

    def test_add_data_compatible(self):
        """Test Argonaut.add_data() with compatible data sets."""

        # first test. "complete" data set
        expected_data_path_1 = os.path.join(arg_data_path, 'Compatible add test 1 result data.txt')
        expected_origin_path_1 = os.path.join(arg_data_path, 'Compatible add test 1 result origin.txt')
        self._test_add_data_compatible(self.data_set_1_path, self.data_set_2_path,
                                       expected_data_path_1, expected_origin_path_1)

        # second test. incomplete data set. configuration parameters are not loaded from a CTL file
        expected_data_path_2 = os.path.join(arg_data_path, 'Compatible add test 2 result data.txt')
        expected_origin_path_2 = os.path.join(arg_data_path, 'Compatible add test 2 result origin.txt')
        self._test_add_data_compatible(self.data_set_3_path, self.data_set_4_path,
                                       expected_data_path_2, expected_origin_path_2)

    def test_add_data_incompatible(self):
        """Test ADVMData.add_data() with incompatible data sets."""
        self._test_add_data_incompatible(self.data_set_1_path, self.data_set_3_path)


if __name__ == '__main__':
    unittest.main()
