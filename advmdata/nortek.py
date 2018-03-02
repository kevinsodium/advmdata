import os
import re

import numpy as np
import pandas as pd

from linearmodel.datamanager import DataManager

from advmdata.core import ADVMData, ADVMConfigParam


def _get_re_value(pattern, string):
    match = re.search(pattern, string)
    return match.group(1)


class AquadoppADVMData(ADVMData):
    """Manages ADVMData for Nortek's Aquadopp instrument"""

    def _calc_cell_range(self):
        """Calculate range of cells along a single beam

        :return:
        """

        # TODO: Implement _calc_first_cell_midpoint() and move most of this up to ADVMData

        blanking_distance = self._configuration_parameters['Blanking Distance']
        cell_size = self._configuration_parameters['Cell Size']
        number_of_cells = self._configuration_parameters['Number of Cells']

        first_cell_mid_point = blanking_distance + cell_size
        last_cell_mid_point = first_cell_mid_point + (number_of_cells - 1) * cell_size

        cell_range = np.linspace(first_cell_mid_point, last_cell_mid_point, num=number_of_cells)

        acoustic_data = self._data_manager.get_data()
        cell_range = np.tile(cell_range, (acoustic_data.shape[0], 1))

        col_names = ['R{:03}'.format(cell) for cell in range(1, number_of_cells+1)]

        cell_range_df = pd.DataFrame(data=cell_range, index=acoustic_data.index, columns=col_names)

        return cell_range_df

    @classmethod
    def _get_whd_header_names(cls, hdr_file_path, whd_file_path):

        # get the name of the whd file
        _, whd_file_name = os.path.split(whd_file_path)

        # read the hdf_file
        with open(hdr_file_path, 'r') as f:
            hdr_lines = f.readlines()

        # find the whd section and read append the lines
        hdr_iter = iter(hdr_lines)
        whd_headers = []
        try:
            while True:
                line = next(hdr_iter)
                if whd_file_name in line:
                    line = next(hdr_iter)
                    while line != '\n':
                        whd_headers.append(line.strip())
                        line = next(hdr_iter)
        except StopIteration:
            pass

        # get the header names
        for i in range(len(whd_headers)):
            split_line = whd_headers[i].split()
            # set the temperature to the standard header
            if 'Temperature' in whd_headers[i]:
                whd_headers[i] = 'Temp'
            # set the noise to the standard header
            elif 'Noise amplitude beam' in whd_headers[i]:
                beam_number = _get_re_value('Noise amplitude beam ([0-9])', whd_headers[i])
                whd_headers[i] = 'Noise' + beam_number
            elif '(' in whd_headers[i]:
                whd_headers[i] = ' '.join(split_line[1:-1])
            else:
                whd_headers[i] = ' '.join(split_line[1:])

        return whd_headers

    @staticmethod
    def _read_a(data_set_path, number_of_cells, beam_number):
        """Read the backscatter amplitude file into a DataFrame

        :param data_set_path:
        :param number_of_cells:
        :param beam_number:
        :return:
        """

        amp_column_names = ['Cell{:02}Amp{:1}'.format(cell, beam_number) for cell in range(1, number_of_cells + 1)]
        amp_file_path = data_set_path + '.a' + str(beam_number)
        amp_df = pd.read_table(amp_file_path, header=None, names=amp_column_names, sep='\s+')

        return amp_df

    @classmethod
    def _read_backscatter(cls, data_set_path, configuration_parameters):
        """Read the multiple backscatter files and return a single DataFrame

        :param data_set_path:
        :param configuration_parameters:
        :return:
        """
        a_list = []
        for beam_number in range(1, configuration_parameters['Number of Beams'] + 1):
            a_df = cls._read_a(data_set_path, configuration_parameters['Number of Cells'], beam_number)
            a_list.append(a_df)

        return pd.concat(a_list, axis=1)

    @staticmethod
    def _read_hdr(data_set_path):
        """Reads the HDR file. Returns configuration parameters.

        :param hdr_file_path:
        :return: config_parameters
        """

        hdr_file_path = data_set_path + '.hdr'

        with open(hdr_file_path, 'r') as f:
            hdr_text = f.read()

        blanking_distance_pattern = 'Blanking distance                     ([0-9]+([.][0-9]*)?|[.][0-9]+) m'
        blanking_distance = float(_get_re_value(blanking_distance_pattern, hdr_text))

        number_of_cells_pattern = 'Number of cells                       ([0-9]*)'
        number_of_cells = int(_get_re_value(number_of_cells_pattern, hdr_text))

        cell_size_pattern = 'Cell size                             ([0-9]+([.][0-9]*)?|[.][0-9]+) cm'
        cell_size = float(_get_re_value(cell_size_pattern, hdr_text)) / 100

        number_of_beams_pattern = 'Number of beams                       ([0-9]*)'
        number_of_beams = int(_get_re_value(number_of_beams_pattern, hdr_text))

        frequency_pattern = 'Head frequency                        ([0-9]+([.][0-9]*)?|[.][0-9]+) kHz'
        frequency = float(_get_re_value(frequency_pattern, hdr_text))

        keys = ['Frequency', 'Beam Orientation', 'Slant Angle', 'Blanking Distance', 'Cell Size', 'Number of Cells',
                'Number of Beams', 'Instrument']
        values = [frequency, 'Horizontal', 25., blanking_distance, cell_size, number_of_cells, number_of_beams, 'AQD']
        config_dict = dict(zip(keys, values))

        configuration_parameters = ADVMConfigParam()
        configuration_parameters.update(config_dict)

        return configuration_parameters

    @classmethod
    def _read_whd(cls, data_set_path):
        """Read the WHD file into a DataFrame

        :param data_set_path:
        :return:
        """

        hdr_file_path = data_set_path + '.hdr'
        whd_file_path = data_set_path + '.whd'

        whd_header_names = cls._get_whd_header_names(hdr_file_path, whd_file_path)

        whd_df = pd.read_table(whd_file_path, header=None, names=whd_header_names, sep='\s+')

        date_time_columns = ['Month', 'Day', 'Year', 'Hour', 'Minute', 'Second']
        datetime_index = pd.to_datetime(whd_df[date_time_columns])

        whd_df.set_index(datetime_index, inplace=True)
        whd_df.drop(date_time_columns, axis=1, inplace=True)

        return whd_df

    @classmethod
    def read_aquadopp_data(cls, data_path, data_set):
        """Read ADVM data from a Nortek Aquadopp data set

        :param data_path: Path to directory containing the Aquadopp data set
        :param data_set: Root name of the files within the data set
        :return:
        """

        data_set_path = os.path.join(data_path, data_set)

        configuration_parameters = cls._read_hdr(data_set_path)
        whd_df = cls._read_whd(data_set_path)
        backscatter_df = cls._read_backscatter(data_set_path, configuration_parameters)
        backscatter_df.set_index(whd_df.index, inplace=True)
        advm_data_df = pd.concat([whd_df, backscatter_df], axis=1)

        advm_data_origin = DataManager.create_data_origin(advm_data_df, data_set_path + "(AQD)")
        advm_data_manager = DataManager(advm_data_df, advm_data_origin)

        return cls(advm_data_manager, configuration_parameters)
