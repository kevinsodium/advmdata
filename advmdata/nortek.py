import os
import re

import numpy as np
import pandas as pd

from linearmodel.datamanager import DataManager
from advmdata.core import ADVMData, ADVMConfigParam, ADVMDataReadError


class NortekADVMData(ADVMData):
    """Super class for Nortek instruments"""

    # Regex patterns for reading config parameters from HDR file
    _cell_size_pattern = 'Cell size                             ([0-9]+([.][0-9]*)?|[.][0-9]+) cm'
    _frequency_pattern = 'Head frequency                        ([0-9]+([.][0-9]*)?|[.][0-9]+) kHz'
    _number_of_beams_pattern = 'Number of beams                       ([0-9]*)'

    # Instrument type, must be defined in subclasses
    _instrument = None

    def _calc_cell_range(self):
        """Calculate range of cells along a single beam

        :return:
        """

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
    def _get_re_value(pattern, string):
        # Search text file for pattern specific to value
        match = re.search(pattern, string)
        return match.group(1)

    @classmethod
    def _read_config_param(cls, data_set_path):
        """

        :param data_set_path: path of data file
        :return: returns dictionary based on hdr file
        """

        # Opens the file path
        hdr_file_path = data_set_path + '.hdr'
        with open(hdr_file_path, 'r') as f:
            hdr_text = f.read()

        number_of_beams = int(cls._get_re_value(cls._number_of_beams_pattern, hdr_text))

        frequency = float(cls._get_re_value(cls._frequency_pattern, hdr_text))

        keys = ['Frequency', 'Beam Orientation', 'Slant Angle', 'Number of Beams',
                'Instrument', 'Effective Transducer Diameter']

        # default values for all(?) Nortek instruments
        nortek_slant_angle = 25.
        nortek_effective_transducer_diameter = 0.01395

        values = [frequency, 'Horizontal', nortek_slant_angle, number_of_beams,
                  cls._instrument, nortek_effective_transducer_diameter]
        config_dict = dict(zip(keys, values))

        configuration_parameters = ADVMConfigParam()
        configuration_parameters.update(config_dict)

        return configuration_parameters


class EZQADVMData(NortekADVMData):
    """Handles specifics for the Nortek EZQ instrument"""

    # Explicit patterns per instrument in HDR file
    _instrument = 'EZQ'
    _number_of_beams_pattern = 'Diagnostics - Number of beams         ([0-9]*)'

    @staticmethod
    def _get_blanking_distance(blanking_distance_series):
        """Gets the blanking distance from the pass Pandas Series.

        Raises an ADVMDataReadError if there are more than one unique blanking distances

        :param blanking_distance_series: Series containing blanking distance
        :return:
        """

        unique_blanking_distance = blanking_distance_series.dropna().unique()

        # make sure there's only one unique blanking distance
        if unique_blanking_distance.shape != (1,):
            raise ADVMDataReadError

        return unique_blanking_distance[0]

    @staticmethod
    def _get_number_of_cells(data_df, number_of_beams):
        """Returns the number of cells based on the number of backscatter columns in data_df

        :param data_df: Pandas DataFrame containing acoustic data
        :param number_of_beams: Number of beams in the instrument configuration
        :return:
        """

        backscatter_regex = r'^(Cell\d{2}Amp\d{1})$'
        backscatter_columns = data_df.filter(regex=backscatter_regex)
        number_of_cells = backscatter_columns.shape[1] / number_of_beams

        if not number_of_cells.is_integer():
            raise ADVMDataReadError("Error reading backscatter data: Unable to determine the number of cells")

        return number_of_cells

    @classmethod
    def _read_backscatter(cls, data_set_path, configuration_parameters):
        """Read the multiple backscatter files and return a single DataFrame

        :param data_set_path: data path
        :param configuration_parameters: config parameters were created in the Super class
        :return: returns the Data frame with a timestamp index
        """

        abs_list = []
        duplicate_columns = ['Blanking', 'Noise1', 'Noise2', 'Noise3', 'Noise4']
        for beam_number in range(1, configuration_parameters['Number of Beams'] + 1):
            beam_abs_df = cls._read_beam_file(data_set_path, beam_number)
            if beam_number != 1:
                beam_abs_df.drop(duplicate_columns, axis=1, inplace=True)
            abs_list.append(beam_abs_df)

        abs_df = pd.concat(abs_list, axis=1)

        return abs_df

    @classmethod
    def _read_beam_file(cls, data_set_path, beam_number):
        """Read a file containing beam data (RA[n])

        :param data_set_path: Root dataset path name
        :param beam_number: Integer indicating the beam number
        :type beam_number: int
        :return: DataFrame containing data in the beam file
        """

        # read the beam file
        beam_file_path = data_set_path + '.ra' + str(beam_number)
        beam_data_df = pd.read_table(beam_file_path, header=None, sep='\s+')

        # add columns names
        beam_data_columns = ['Month', 'Day', 'Year', 'Hour', 'Minute', 'Second', 'Blanking',
                             'Noise1', 'Noise2', 'Noise3', 'Noise4']
        number_of_cells = beam_data_df.shape[1] - len(beam_data_columns)
        amp_columns = ['Cell{0:02}Amp{1:1}'.format(cell, beam_number) for cell in range(1, number_of_cells + 1)]
        beam_data_columns.extend(amp_columns)
        beam_data_df.columns = beam_data_columns

        # convert date/time columns to DateTimeIndex and drop the date/time columns
        date_time_columns = ['Month', 'Day', 'Year', 'Hour', 'Minute', 'Second']
        date_time_index = pd.to_datetime(beam_data_df[date_time_columns])
        beam_data_df.set_index(date_time_index, inplace=True)
        beam_data_df.drop(date_time_columns, axis=1, inplace=True)

        return beam_data_df

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

    @staticmethod
    def _read_sen_file(data_set_path):
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
        vertical_beam.columns = ['Vbeam']

        return vertical_beam

    @classmethod
    def read_ezq_data(cls, data_set_path, cell_size):
        """

        :param data_set_path:
        :param cell_size:
        :return:
        """

        # Create config parameters
        configuration_parameters = cls._read_config_param(data_set_path)
        configuration_parameters['Cell Size'] = cell_size

        # Create DataFrames
        v_beam_df = cls._read_sen_file(data_set_path)
        temp_df = cls._read_dat_file(data_set_path)
        amp_df = cls._read_backscatter(data_set_path, configuration_parameters)

        # Merge all Data Frames
        data_df = pd.concat([v_beam_df, temp_df, amp_df], axis=1)

        blanking_series = data_df['Blanking']
        blanking_distance = cls._get_blanking_distance(blanking_series)
        configuration_parameters['Blanking Distance'] = blanking_distance

        number_of_cells = cls._get_number_of_cells(data_df, configuration_parameters['Number of Beams'])
        configuration_parameters['Number of Cells'] = int(number_of_cells)

        # create DataManager
        advm_data_origin = DataManager.create_data_origin(data_df, data_set_path + " (EZQ)")
        advm_data_manager = DataManager(data_df, advm_data_origin)

        return cls(advm_data_manager, configuration_parameters)


class AquadoppADVMData(NortekADVMData):
    """Handles specifics for the Nortek Aquadopp instrument"""

    _instrument = 'AQD'

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
    def _get_blanking_distance(cls, data_set_path):
        """Reads the blanking distance from the HDR file

        :param data_set_path: Path name of the root dataset
        :return:
        """

        # Opens the file path
        hdr_file_path = data_set_path + '.hdr'
        with open(hdr_file_path, 'r') as f:
            hdr_text = f.read()

        blanking_distance_pattern = 'Blanking distance                     ([0-9]+([.][0-9]*)?|[.][0-9]+) m'
        blanking_distance = float(cls._get_re_value(blanking_distance_pattern, hdr_text))

        return blanking_distance

    @classmethod
    def _read_number_of_cells(cls, data_set_path):

        # Read the number of cells from HDR file, specific to AQD instrument
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

        configuration_parameters = cls._read_config_param(data_set_path)

        sen_df = cls._read_time_series_file(data_set_path, '.sen')

        whd_df = cls._read_time_series_file(data_set_path, '.whd')
        # whd_df = cls._combine_time_index(whd_df, sen_df)
        whd_df.drop(labels=['Temp'], axis=1, inplace=True)

        backscatter_df = cls._read_backscatter(data_set_path, configuration_parameters)
        backscatter_df.set_index(sen_df.index, inplace=True)
        advm_data_df = pd.concat([sen_df, whd_df, backscatter_df], axis=1)

        advm_data_origin = DataManager.create_data_origin(advm_data_df, data_set_path + "(AQD)")
        advm_data_manager = DataManager(advm_data_df, advm_data_origin)

        return cls(advm_data_manager, configuration_parameters)
