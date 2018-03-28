import abc
import os
import re

import numpy as np
import pandas as pd
import linecache

from linearmodel.linearmodel.datamanager import DataManager
from advmdata.advmdata.core import ADVMData, ADVMConfigParam


class NortekADVMData(ADVMData):
    _blanking_distance_pattern = None
    _cell_size_pattern = None
    _frequency_pattern = None
    _instrument = None
    _number_of_beams_pattern = None

    @classmethod
    def combine_time_index(cls, initial_df, wanted_index_df):
        """

        :param initial_df: DataFrame with unaligned index
        :param wanted_index_df: DataFrame with desired index
        :return: DataFrame with new aligned index
        """

        # Store original index
        initial_df_original_index = initial_df.index

        # Append new index to initial Data Frame
        initial_df = initial_df.append(pd.DataFrame(index=wanted_index_df.index))

        # Interpolate and drop original index
        initial_df.sort_index(inplace=True)
        initial_df.interpolate(inplace=True)
        initial_df.drop(labels=initial_df_original_index, axis=0, inplace=True)

        return initial_df

    @staticmethod
    def _get_re_value(pattern, string):
        # Search text file for pattern specific to value
        match = re.search(pattern, string)
        return match.group(1)

    @staticmethod
    @abc.abstractstaticmethod
    def _read_number_of_cells(data_set_path):
        pass

    @classmethod
    def read_config_param(cls, data_set_path):
        """

        :param data_set_path: path of data file
        :return: returns dictionary based on hdr file
        """

        # Try two patterns for ADQ and EZQ
        hdr_file_path = data_set_path + '.hdr'
        with open(hdr_file_path, 'r') as f:
            hdr_text = f.read()

        blanking_distance = float(cls._get_re_value(cls._blanking_distance_pattern, hdr_text))

        number_of_cells = cls._read_number_of_cells(data_set_path)

        cell_size = float(cls._get_re_value(cls._cell_size_pattern, hdr_text)) / 100

        number_of_beams = int(cls._get_re_value(cls._number_of_beams_pattern, hdr_text))

        frequency = float(cls._get_re_value(cls._frequency_pattern, hdr_text))

        keys = ['Frequency', 'Beam Orientation', 'Slant Angle', 'Blanking Distance', 'Cell Size', 'Number of Cells',
                'Number of Beams', 'Instrument', 'Effective Transducer Diameter']

        values = [frequency, 'Horizontal', 25., blanking_distance, cell_size, number_of_cells,
                  number_of_beams, cls._instrument, 0.01395]
        config_dict = dict(zip(keys, values))

        configuration_parameters = ADVMConfigParam()
        configuration_parameters.update(config_dict)

        return configuration_parameters


class EzqADVMData(NortekADVMData):
    def __init__(self):
        super(EzqADVMData, self).__init__()

    _blanking_distance_pattern = 'Blanking distance                     ([0-9]+([.][0-9]*)?|[.][0-9]+) m'
    _cell_size_pattern = 'Cell size                             ([0-9]+([.][0-9]*)?|[.][0-9]+) cm'
    _frequency_pattern = 'Head frequency                        ([0-9]+([.][0-9]*)?|[.][0-9]+) kHz'
    _instrument = 'EZQ'
    _number_of_beams_pattern = 'Diagnostics - Number of beams         ([0-9]*)'

    def _calc_cell_range(self):
        """Calculate range of cells along a single beam

        :return:
        """

        # TODO: Implement _calc_first_cell_midpoint() and move most of this up to ADVMData

        blanking_distance = self._configuration_parameters['Blanking Distance']
        cell_size = self._configuration_parameters['Cell Size']
        number_of_cells = self._configuration_parameters['Number of Cells']

        first_cell_mid_point = blanking_distance + cell_size / 2
        last_cell_mid_point = first_cell_mid_point + (number_of_cells - 1) * cell_size

        cell_range = np.linspace(first_cell_mid_point, last_cell_mid_point, num=number_of_cells)

        acoustic_data = self._data_manager.get_data()
        cell_range = np.tile(cell_range, (acoustic_data.shape[0], 1))

        col_names = ['R{:03}'.format(cell) for cell in range(1, number_of_cells + 1)]

        cell_range_df = pd.DataFrame(data=cell_range, index=acoustic_data.index, columns=col_names)

        return cell_range_df

    @staticmethod
    def _read_number_of_cells(data_set_path):
        data_file_path = data_set_path + '.ra1'
        line = linecache.getline(data_file_path, 1).split("   ")
        number_of_cells = len(line) - 6

        return number_of_cells

    @staticmethod
    def _read_amp_from_ra(data_set_path, number_of_cells, beam_number):
        """Read the backscatter amplitude file into a DataFrame

        :param data_set_path:
        :param number_of_cells:
        :param beam_number:
        :return:
        """
        amp_column_names = ['Cell{:02}Amp{:1}'.format(cell, beam_number) for cell in range(1, number_of_cells + 1)]
        amp_file_path = data_set_path + '.ra' + str(beam_number)
        amp_df = pd.read_table(amp_file_path, header=None, sep='\s+')
        amp_df = amp_df.drop(amp_df.columns[0:11], axis=1)
        amp_df.columns = amp_column_names

        return amp_df

    @classmethod
    def _read_backscatter(cls, data_set_path, configuration_parameters):
        """Read the multiple backscatter files and return a single DataFrame

        :param data_set_path:
        :param configuration_parameters:
        :return:
        """
        datetime_index = cls._read_time_stamp(data_set_path)
        a_list = []
        for beam_number in range(1, configuration_parameters['Number of Beams'] + 1):
            a_df = cls._read_amp_from_ra(data_set_path, configuration_parameters['Number of Cells'], beam_number)
            a_list.append(a_df)

        a_list = pd.concat(a_list, axis=1)
        a_list = a_list.set_index(datetime_index)

        return a_list

    @staticmethod
    def _read_time_stamp(data_set_path):
        """Read a time stamps of a file into a DataFrame

        :param data_set_path:
        :param data_file_suffix:
        :return:
        """
        data_file_path = data_set_path + '.ra1'
        time_df = pd.read_table(data_file_path, header=None, sep='\s+')
        time_df = time_df.drop(time_df.columns[6:], axis=1)

        date_time_columns = ['Month', 'Day', 'Year', 'Hour', 'Minute', 'Second']
        time_df.columns = date_time_columns

        datetime_index = pd.to_datetime(time_df)

        return datetime_index

    @staticmethod
    def _read_sens_file(data_set_path):
        """Read the vertical beam data into a DataFrame

        :param data_set_path:
        :return:
        """
        data_file_path = data_set_path + '.sen'
        time_df = pd.read_table(data_file_path, header=None, sep='\s+')
        time_df = time_df.drop(time_df.columns[6:], axis=1)

        date_time_columns = ['Month', 'Day', 'Year', 'Hour', 'Minute', 'Second']
        time_df.columns = date_time_columns

        datetime_index = pd.to_datetime(time_df)

        sens_file = pd.read_table(data_file_path, header=None, sep='\s+')
        vertical_beam = pd.DataFrame(sens_file[10])
        vertical_beam.index = datetime_index
        vertical_beam.columns = ['Vertical Beam']

        return vertical_beam

    @staticmethod
    def _read_dat_file(data_set_path):
        """Read the Temperature data into a DataFrame

        :param data_set_path:
        :return:
        """

        data_file_path = data_set_path + '.dat'
        time_df = pd.read_table(data_file_path, header=None, sep='\s+')
        time_df = time_df.drop(time_df.columns[6:], axis=1)

        date_time_columns = ['Month', 'Day', 'Year', 'Hour', 'Minute', 'Second']
        time_df.columns = date_time_columns

        datetime_index = pd.to_datetime(time_df)
        dat_file = pd.read_table(data_file_path, header=None, sep='\s+')
        temperature = pd.DataFrame(dat_file[10])
        temperature.index = datetime_index
        temperature.columns = ['Temp']

        return temperature

    @classmethod
    def create_df(cls, data_file_path, data_set):
        """

        :param data_file_path:
        :param data_set:
        :return:
        """
        data_set_path = os.path.join(data_file_path, data_set)

        v_beam_df = cls._read_sens_file(data_set_path)
        temp_df = cls._read_dat_file(data_set_path)
        configuration_parameters = cls.read_config_param(data_set_path)

        amp_df = cls._read_backscatter(data_set_path, configuration_parameters)

        data_df = pd.concat([v_beam_df, temp_df], axis=1)

        data_df = cls.combine_time_index(data_df, amp_df)

        advm_data_df = pd.concat([data_df, amp_df], axis=1)
        advm_data_origin = DataManager.create_data_origin(advm_data_df, data_set_path + "(EzQ)")

        advm_data_manager = DataManager(advm_data_df, advm_data_origin)

        return cls(advm_data_manager, configuration_parameters)


class AquadoppADVMData(NortekADVMData):
    """Manages ADVMData for Nortek's Aquadopp instrument"""
    def __init__(self):
        super(AquadoppADVMData, self).__init__()

    _cell_size_pattern = 'Cell size                             ([0-9]+([.][0-9]*)?|[.][0-9]+) cm'
    _blanking_distance_pattern = 'Blanking distance                     ([0-9]+([.][0-9]*)?|[.][0-9]+) m'
    _frequency_pattern = 'Head frequency                        ([0-9]+([.][0-9]*)?|[.][0-9]+) kHz'
    _instrument = 'AQD'
    _number_of_beams_pattern = 'Number of beams                       ([0-9]*)'

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

        col_names = ['R{:03}'.format(cell) for cell in range(1, number_of_cells + 1)]

        cell_range_df = pd.DataFrame(data=cell_range, index=acoustic_data.index, columns=col_names)

        return cell_range_df

    @classmethod
    def _get_header_names_from_hdr(cls, hdr_file_path, data_file_path):

        # get the name of the whd file
        _, data_file_name = os.path.split(data_file_path)

        # read the hdf_file
        with open(hdr_file_path, 'r') as f:
            hdr_lines = f.readlines()

        # find the whd section and read append the lines
        hdr_iter = iter(hdr_lines)
        header_names = []
        try:
            while True:
                line = next(hdr_iter)
                if data_file_name in line:
                    line = next(hdr_iter)
                    while line != '\n':
                        header_names.append(line.strip())
                        line = next(hdr_iter)
        except StopIteration:
            pass

        # get the header names
        for i in range(len(header_names)):
            split_line = header_names[i].split()
            # set the temperature to the standard header
            if 'Temperature' in header_names[i]:
                header_names[i] = 'Temp'
            # set the noise to the standard header
            elif 'Noise amplitude beam' in header_names[i]:
                beam_number = cls._get_re_value('Noise amplitude beam ([0-9])', header_names[i])
                header_names[i] = 'Noise' + beam_number
            elif '(' in header_names[i]:
                header_names[i] = ' '.join(split_line[1:-1])
            else:
                header_names[i] = ' '.join(split_line[1:])

        return header_names

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

    @classmethod
    def _read_number_of_cells(cls, data_set_path):

        hdr_file_path = data_set_path + '.hdr'
        with open(hdr_file_path, 'r') as f:
            hdr_text = f.read()

        number_of_cells_pattern = 'Number of cells                       ([0-9]*)'

        number_of_cells = int(cls._get_re_value(number_of_cells_pattern, hdr_text))

        return number_of_cells

    @classmethod
    def _read_time_series_file(cls, data_set_path, data_file_suffix):
        """Read a time series file into a DataFrame

        :param data_set_path:
        :param data_file_suffix:
        :return:
        """

        hdr_file_path = data_set_path + '.hdr'
        data_file_path = data_set_path + data_file_suffix

        header_names = cls._get_header_names_from_hdr(hdr_file_path, data_file_path)

        data_file_df = pd.read_table(data_file_path, header=None, names=header_names, sep='\s+')

        date_time_columns = ['Month', 'Day', 'Year', 'Hour', 'Minute', 'Second']
        datetime_index = pd.to_datetime(data_file_df[date_time_columns])

        data_file_df.set_index(datetime_index, inplace=True)
        data_file_df.drop(date_time_columns, axis=1, inplace=True)

        return data_file_df

    @classmethod
    def read_aquadopp_data(cls, data_path, data_set):
        """Read ADVM data from a Nortek Aquadopp data set

        :param data_path: Path to directory containing the Aquadopp data set
        :param data_set: Root name of the files within the data set
        :return:
        """

        data_set_path = os.path.join(data_path, data_set)

        configuration_parameters = cls.read_config_param(data_set_path)
        number_of_cells = cls._read_number_of_cells(data_set_path)
        configuration_parameters['Number of Cells'] = number_of_cells

        sen_df = cls._read_time_series_file(data_set_path, '.sen')

        whd_df = cls._read_time_series_file(data_set_path, '.whd')
        whd_df = cls.combine_time_index(whd_df, sen_df)
        whd_df.drop(labels=['Temp'], axis=1, inplace=True)

        backscatter_df = cls._read_backscatter(data_set_path, configuration_parameters)
        backscatter_df.set_index(sen_df.index, inplace=True)
        advm_data_df = pd.concat([sen_df, whd_df, backscatter_df], axis=1)

        advm_data_origin = DataManager.create_data_origin(advm_data_df, data_set_path + "(AQD)")
        advm_data_manager = DataManager(advm_data_df, advm_data_origin)

        return cls(advm_data_manager, configuration_parameters)

