import abc
import unittest

import numpy as np
import pandas as pd


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
