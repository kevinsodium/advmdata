import os

import numpy as np
import pandas as pd
import scipy
from datetime import datetime
from datetime import timedelta
import linecache

from scipy import io

from linearmodel import datamanager

from advmdata.core import ADVMData, ADVMConfigParam


class ArgonautADVMData(ADVMData):
    """Instrument-specific data for the SonTek Argonaut SL (second generation SL)"""

    def _calc_cell_range(self):
        """Calculate range of cells along a single beam.

        :return: Range of cells along a single beam
        """

        blanking_distance = self._configuration_parameters['Blanking Distance']
        cell_size = self._configuration_parameters['Cell Size']
        number_of_cells = self._configuration_parameters['Number of Cells']

        first_cell_mid_point = blanking_distance + cell_size / 2
        last_cell_mid_point = first_cell_mid_point + (number_of_cells - 1) * cell_size

        cell_range = np.linspace(first_cell_mid_point, last_cell_mid_point, num=number_of_cells)

        acoustic_data = self._data_manager.get_data()
        cell_range = np.tile(cell_range, (acoustic_data.shape[0], 1))

        col_names = ['R{:03}'.format(cell) for cell in range(1, number_of_cells+1)]

        cell_range_df = pd.DataFrame(data=cell_range, index=acoustic_data.index, columns=col_names)

        return cell_range_df

    @staticmethod
    def _read_argonaut_ctl_file(arg_ctl_filepath):
        """
        Read the Argonaut '.ctl' file into a configuration dictionary.

        :param arg_ctl_filepath: Filepath containing the Argonaut '.dat' file
        :return: Dictionary containing specific configuration parameters
        """

        if not os.path.isfile(arg_ctl_filepath):
            raise FileNotFoundError('{0} does not exist'.format(arg_ctl_filepath))

        # Read specific configuration values from the Argonaut '.ctl' file into a dictionary.
        # The fixed formatting of the '.ctl' file is leveraged to extract values from foreknown file lines.
        config_dict = {}
        line = linecache.getline(arg_ctl_filepath, 10).strip()
        arg_type = line.split("ArgType ------------------- ")[-1:]

        if arg_type == "SL":
            config_dict['Beam Orientation'] = "Horizontal"
            config_dict['Instrument'] = 'SL'
        else:
            config_dict['Beam Orientation'] = "Vertical"
            config_dict['Instrument'] = 'SW'

        line = linecache.getline(arg_ctl_filepath, 12).strip()
        frequency = line.split("Frequency ------- (kHz) --- ")[-1:]
        config_dict['Frequency'] = float(frequency[0])

        # calculate transducer radius (m)
        if float(frequency[0]) == 3000:
            config_dict['Effective Transducer Diameter'] = 0.015
        elif float(frequency[0]) == 1500:
            config_dict['Effective Transducer Diameter'] = 0.030
        elif float(frequency[0]) == 500:
            config_dict['Effective Transducer Diameter'] = 0.090
        elif np.isnan(float(frequency[0])):
            config_dict['Effective Transducer Diameter'] = "NaN"

        config_dict['Number of Beams'] = int(2)  # always 2; no need to check file for value

        line = linecache.getline(arg_ctl_filepath, 16).strip()
        slant_angle = line.split("SlantAngle ------ (deg) --- ")[-1:]
        config_dict['Slant Angle'] = float(slant_angle[0])

        line = linecache.getline(arg_ctl_filepath, 44).strip()
        slant_angle = line.split("BlankDistance---- (m) ------ ")[-1:]
        config_dict['Blanking Distance'] = float(slant_angle[0])

        line = linecache.getline(arg_ctl_filepath, 45).strip()
        cell_size = line.split("CellSize -------- (m) ------ ")[-1:]
        config_dict['Cell Size'] = float(cell_size[0])

        line = linecache.getline(arg_ctl_filepath, 46).strip()
        number_cells = line.split("Number of Cells ------------ ")[-1:]
        config_dict['Number of Cells'] = int(number_cells[0])

        return config_dict

    @staticmethod
    def _read_argonaut_dat_file(arg_dat_filepath):
        """
        Read the Argonaut '.dat' file into a DataFrame.

        :param arg_dat_filepath: Filepath containing the Argonaut '.dat' file
        :return: Timestamp formatted DataFrame containing '.dat' file contents
        """

        # Read the Argonaut '.dat' file into a DataFrame
        dat_df = pd.read_table(arg_dat_filepath, sep='\s+')

        # rename the relevant columns to the standard/expected names
        dat_df.rename(columns={"Temperature": "Temp", "Level": "Vbeam"}, inplace=True)

        # set dataframe index by using date/time information
        date_time_columns = ["Year", "Month", "Day", "Hour", "Minute", "Second"]
        datetime_index = pd.to_datetime(dat_df[date_time_columns])
        dat_df.set_index(datetime_index, inplace=True)

        # remove non-relevant columns
        relevant_columns = ['Temp', 'Vbeam', 'Velocity(X|Y)']
        dat_df = dat_df.filter(regex=r'(' + '|'.join(relevant_columns) + r')$')

        dat_df = dat_df.apply(pd.to_numeric, args=('coerce', ))
        dat_df = dat_df.astype(np.float)

        return dat_df

    @staticmethod
    def _read_argonaut_multicell_file(arg_snr_filepath, column_regex=None):
        """
        Read Argonaut multi-cell files (vel & snr) into a DataFrame.

        :param arg_snr_filepath: Filepath containing the Argonaut '.snr' or '.vel' file
        :param column_regex: Regular expression for relevant columns
        :return: Timestamp formatted DataFrame containing '.snr' or '.vel' file contents
        """

        # Read the Argonaut '.snr' file into a DataFrame, combine first two rows to make column headers,
        # and remove unused datetime columns from the DataFrame.
        snr_df = pd.read_table(arg_snr_filepath, sep='\s+', header=None)
        header = snr_df.ix[0] + snr_df.ix[1]
        snr_df.columns = header.str.replace(r"\(.*\)", "")  # remove parentheses and everything inside them from headers
        snr_df = snr_df.ix[2:]

        # rename columns to recognizable date/time elements
        column_names = list(snr_df.columns)
        column_names[1] = 'Year'
        column_names[2] = 'Month'
        column_names[3] = 'Day'
        column_names[4] = 'Hour'
        column_names[5] = 'Minute'
        column_names[6] = 'Second'
        snr_df.columns = column_names

        # create a datetime index and set the dataframe index
        datetime_index = pd.to_datetime(snr_df.ix[:, 'Year':'Second'])
        snr_df.set_index(datetime_index, inplace=True)

        # remove non-relevant columns
        if column_regex is not None:
            snr_df = snr_df.filter(regex=column_regex)

        snr_df = snr_df.apply(pd.to_numeric, args=('coerce', ))
        snr_df = snr_df.astype(np.float)

        return snr_df

    @classmethod
    def read_argonaut_data(cls, data_directory, filename):
        """Loads an Argonaut data set into an ADVMData class object.

        The DAT, SNR, and CTL ASCII files that are exported (with headers) from ViewArgonaut must be present.

        :param data_directory: file path containing the Argonaut data files
        :type data_directory: str
        :param filename: root filename for the 3 Argonaut files
        :type filename: str
        :return: ADVMData object containing the Argonaut data set information
        """

        dataset_path = os.path.join(data_directory, filename)

        # Read the Argonaut '.dat' file into a DataFrame
        arg_dat_file = dataset_path + ".dat"
        dat_df = cls._read_argonaut_dat_file(arg_dat_file)

        # Read the Argonaut '.snr' file into a DataFrame
        arg_snr_file = dataset_path + ".snr"
        try:
            snr_df = cls._read_argonaut_multicell_file(arg_snr_file, cls._advm_columns_regex)
        except FileNotFoundError:
            snr_df = pd.DataFrame()

        arg_vel_file = dataset_path + ".vel"
        try:
            vel_df = cls._read_argonaut_multicell_file(arg_vel_file, cls._advm_columns_regex)
        except FileNotFoundError:
            vel_df = pd.DataFrame()

        # Read specific configuration values from the Argonaut '.ctl' file into a dictionary.
        arg_ctl_file = dataset_path + ".ctl"
        configuration_parameters = ADVMConfigParam()
        try:
            config_dict = cls._read_argonaut_ctl_file(arg_ctl_file)
            configuration_parameters.update(config_dict)
        except FileNotFoundError:
            pass

        # Combine the '.snr' and '.dat.' DataFrames into a single acoustic DataFrame, make the timestamp
        # the index, and return an ADVMData object
        acoustic_df = pd.concat([dat_df, snr_df, vel_df], axis=1)
        data_origin = datamanager.DataManager.create_data_origin(acoustic_df, dataset_path + "(Arg)")

        data_manager = datamanager.DataManager(acoustic_df, data_origin)

        return cls(data_manager, configuration_parameters)


class SL3GADVMData(ADVMData):

    def _calc_cell_range(self):
        """Calculate range of cells along a single beam.

        :return: Range of cells along a single beam
        """
        acoustic_data = self._data_manager.get_data()

        # Finds config dictionary definitions and determines cell sizes
        blanking_distance = self._configuration_paramters['Blanking Distance']
        cell_size = self._configuration_parameters['Cell Size']
        number_of_cells = self._configuration_parameters['Number of Cells']

        # 3G is blanking distance + 1.5* cell size
        first_cell_midpoint = blanking_distance + 1.5*cell_size
        last_cell_mid_point = first_cell_midpoint + (number_of_cells - 1)*cell_size

        # defines linspace
        cell_range = np.linspace(first_cell_midpoint, last_cell_mid_point, num = number_of_cells)
        cell_range = np.tile(cell_range, (acoustic_data.shape[0], 1))

        col_names = ['R{:03}'.format(cell) for cell in range(1, number_of_cells + 1)]
        cell_range_df = pd.DataFrame(data=cell_range, index=acoustic_data.index, columns =col_names)

        return cell_range_df

    @staticmethod
    def _read_mat_file_param(sontek_file_path):
        """
        Read the Sontek Mat file into a configuration dictionary

        :param sontek_file_path: Filepath containing the Sontek '.mat' file
        :return: Dictionary containing specific configuration parameters
        """

        # Initialize the matfile into a variable
        mat_file = scipy.io.loadmat(sontek_file_path, struct_as_record=True, squeeze_me=False)
        config_dict = {}

        # Determine the instrument type
        instrument_type = mat_file['System_Id'][0, 0]['InstrumentType']
        if 'IQ' in instrument_type[0]:
            config_dict['Instrument'] = "IQ"
        else:
            config_dict['Instrument'] = "SL"

        # Match instrument type to Frequency
        if config_dict['Instrument'] == "IQ":
            config_dict['Frequency'] = 3000
        # Determine Frequency
        else:
            config_dict['Frequency'] = 3000

        # Default Slant Angle
        config_dict['Slant Angle'] = 25

        # Default Number of Beams
        config_dict['Number of Beams'] = int(2)

        # Read Blanking Distance in as CM, return M
        blanking_dist = mat_file['System_IqSetup'][0, 0]['advancedSetup']['SLblankingDistance']
        if blanking_dist > 100:
            blanking_dist = blanking_dist / 100
        config_dict['Blanking Distance'] = float(blanking_dist[0])

        # Read Cell Size in as CM, return M
        cell_size = mat_file['System_IqSetup'][0, 0]['advancedSetup']['SLcellSize']
        # TODO: Check the units to see if we need to convert to meters
        if cell_size > 100:
            cell_size = cell_size/100
        config_dict['Cell Size'] = float(cell_size[0])

        # Number of Cells
        number_cells = mat_file['System_IqSetup'][0, 0]['advancedSetup']['SLcellCount']
        config_dict['Number of Cells'] = int(number_cells[0])

        # Return created configuration dictionary
        return config_dict

    @staticmethod
    def _read_mat_file_dat(sontek_file_path):
        """
        Read the '.mat' file and create a DataFrame for flow parameters

        :param sontek_file_path: Filepath containing the Sontek '.mat' file
        :return: Timestamp formatted DataFrame containing flow parameters
        """
        # Load .mat file into a variable
        mat_file = scipy.io.loadmat(sontek_file_path, struct_as_record=True, squeeze_me=False)

        # Read sample time in as Serial Number (in microseconds from Jan 1st, 2000) and give it a timestamp
        sample_time = mat_file['FlowData_SampleTime']
        sample_time = sample_time.flatten()
        sample_times = [datetime(2000, 1, 1, 0, 0) + timedelta(microseconds=x) for x in sample_time]

        # Set index as the timestamp
        datetime_index = pd.to_datetime(sample_times)

        # Read in sample number total
        sample_number = mat_file['FlowData_SampleNumber']
        sample_number = sample_number.flatten()

        # Read in temperature
        temperature = mat_file['FlowData_Temp']
        temperature = temperature.flatten()

        # Read in vertical beam distance from Depth
        v_beam = mat_file['FlowData_Depth']
        v_beam = v_beam.flatten()

        # Create a DataFrame from variables above with 'index=timestamp'
        dat_df = pd.DataFrame({'Temp': temperature, 'Vertical Beam': v_beam}, index=sample_number)
        dat_df.set_index(datetime_index, inplace=True)

        return dat_df

    @staticmethod
    def _read_sontek_snr(sontek_file_path):
        """
        Read in .mat file and create a DataFrame with both beam's SNR data

        :param sontek_file_path: sontek_file_path: Filepath containing the Sontek '.mat' file
        :return: DataFrame for both beams SNR data
        """

        # Load .mat file into a variable
        mat_file = scipy.io.loadmat(sontek_file_path, struct_as_record=True, squeeze_me=False)

        # Intensity factor not yet give
        intensity_factor = np.NaN

        # Beam one SNR data
        beam_one_amp = mat_file['Profile_0_Amp']
        beam_snr = mat_file['FlowData_SNR']
        beam_one_snr = intensity_factor * (beam_one_amp - beam_snr[:, 0, None])

        # Beam two SNR data
        beam_two_amp = mat_file['Profile_1_Amp']
        beam_snr = mat_file['FlowData_SNR']
        beam_two_snr = intensity_factor * (beam_two_amp - beam_snr[:, 1, None])

        # Create DataFrames for both beams and merge them
        snr_df1 = pd.DataFrame(beam_one_snr)
        snr_df2 = pd.DataFrame(beam_two_snr)
        snr_df = pd.concat([snr_df1, snr_df2], axis=1)

        return snr_df

    @classmethod
    def read_sl3g_mat(cls, mat_file_path):
        """
        Loads the Sontek Data into the ADVM Data class object, retrieves data, snr and paramaters
        with cell location

        :param mat_file_path: file path containing the .mat file for the sontek data
        :return: ADVMData object containing the Sontek Data
        """

        # Read the Argonaut '.mat' files into DataFrame
        dat_df = cls._read_mat_file_dat(mat_file_path)
        snr_df = cls._read_sontek_snr(mat_file_path)
        config_dict = cls._read_mat_file_param(mat_file_path)

        # Read specific configuration values from '.mat' file
        configuration_parameters = ADVMConfigParam()
        configuration_parameters.update(config_dict)

        # Combine both DataFrames
        acoustic_df = pd.concat([dat_df, snr_df], axis=1)
        data_origin = datamanager.DataManager.create_data_origin(acoustic_df, dataset_path + "(Sontek")

        data_manager = datamanager.DataManager(acoustic_df, data_origin)

        return cls(data_manager, configuration_parameters)
