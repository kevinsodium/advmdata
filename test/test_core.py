import abc
import unittest

import numpy as np
import pandas as pd

from advmdata.core import ADVMDataIncompatibleError


class TestADVMDataInit(unittest.TestCase):

    advm_data = None
    data_set_path = None
    expected_config_dict = None

    @abc.abstractmethod
    def _get_expected_cell_range_data(self):
        pass

    def _test_cell_range(self):
        """Tests the calculation of the cell range"""
        calculated_cell_range = self.advm_data.get_cell_range().mean()

        number_of_cells = self.expected_config_dict['Number of Cells']
        cell_numbers = np.arange(1, number_of_cells + 1)
        expected_cell_range_index = ['R{:03}'.format(cell) for cell in cell_numbers]

        # create a Series with the expected cell ranges
        expected_cell_range_data = self._get_expected_cell_range_data()
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

        data_path_suffix = " (" + self.expected_config_dict['Instrument'] + ")"
        expected_origin.replace(to_replace='{origin path}', value=self.data_set_path + data_path_suffix, inplace=True)

        pd.testing.assert_frame_equal(read_results_origin, expected_origin)


class TestADVMDataAddData(unittest.TestCase):

    read_data_method = None

    def _test_add_data_compatible(self, data_set_1_path, data_set_2_path, expected_data_path, expected_origin_path):
        """Test the add_data method with compatible data sets."""

        advm_data_1 = self.read_data_method(data_set_1_path)
        advm_data_2 = self.read_data_method(data_set_2_path)

        advm_data_add_result = advm_data_1.add_data(advm_data_2)

        # test the resulting data of the add
        result_data_df = advm_data_add_result.get_data()
        expected_data_df = pd.read_table(expected_data_path, index_col=0, parse_dates=True)
        pd.testing.assert_frame_equal(result_data_df, expected_data_df)

        # test the resulting origin of the add
        result_origin_df = advm_data_add_result.get_origin()
        expected_origin_df = pd.read_table(expected_origin_path, index_col=0)
        result_configuration_parameters = advm_data_add_result.get_configuration_parameters()
        expected_instrument_type = result_configuration_parameters['Instrument']
        origin_path_1 = data_set_1_path + " (" + expected_instrument_type + ")"
        origin_path_2 = data_set_2_path + " (" + expected_instrument_type + ")"
        expected_origin_df.replace(to_replace=['{origin path 1}', '{origin path 2}'],
                                   value=[origin_path_1, origin_path_2], inplace=True)
        pd.testing.assert_frame_equal(result_origin_df, expected_origin_df)

    def _test_add_data_incompatible(self, data_set_1_path, data_set_2_path):
        """Test the add_data method with incompatible data sets."""
        advm_data_1 = self.read_data_method(data_set_1_path)
        advm_data_2 = self.read_data_method(data_set_2_path)
        self.assertRaises(ADVMDataIncompatibleError, advm_data_1.add_data, advm_data_2)
