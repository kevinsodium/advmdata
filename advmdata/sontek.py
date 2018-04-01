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

        config_dict['Instrument'] = arg_type[0]

        if config_dict['Instrument'] == 'SL':
            config_dict['Beam Orientation'] = "Horizontal"
        elif config_dict['Instrument'] == 'SW':
            config_dict['Beam Orientation'] = "Vertical"

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
        header = snr_df.iloc[0] + snr_df.iloc[1]
        snr_df.columns = header.str.replace(r"\(.*\)", "")  # remove parentheses and everything inside them from headers
        snr_df = snr_df.iloc[2:]

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
        datetime_index = pd.to_datetime(snr_df.loc[:, 'Year':'Second'])
        snr_df.set_index(datetime_index, inplace=True)

        # remove non-relevant columns
        if column_regex is not None:
            snr_df = snr_df.filter(regex=column_regex)

        snr_df = snr_df.apply(pd.to_numeric, args=('coerce', ))
        snr_df = snr_df.astype(np.float)

        return snr_df

    @classmethod
    def read_argonaut_data(cls, data_set_path):
        """Loads an Argonaut data set into an ADVMData class object.

        The DAT, SNR, and CTL ASCII files that are exported (with headers) from ViewArgonaut must be present.

        :param data_directory: Full path to the Argonaut data files
        :type data_directory: str
        :return: ADVMData object containing the Argonaut data set information
        """

        # Read the Argonaut '.dat' file into a DataFrame
        arg_dat_file = data_set_path + ".dat"
        dat_df = cls._read_argonaut_dat_file(arg_dat_file)

        # Read the Argonaut '.snr' file into a DataFrame
        arg_snr_file = data_set_path + ".snr"
        try:
            snr_df = cls._read_argonaut_multicell_file(arg_snr_file, cls._advm_columns_regex)
        except FileNotFoundError:
            snr_df = pd.DataFrame()

        arg_vel_file = data_set_path + ".vel"
        try:
            vel_df = cls._read_argonaut_multicell_file(arg_vel_file, cls._advm_columns_regex)
        except FileNotFoundError:
            vel_df = pd.DataFrame()

        # Read specific configuration values from the Argonaut '.ctl' file into a dictionary.
        arg_ctl_file = data_set_path + ".ctl"
        configuration_parameters = ADVMConfigParam()
        try:
            config_dict = cls._read_argonaut_ctl_file(arg_ctl_file)
            configuration_parameters.update(config_dict)
        except FileNotFoundError:
            pass

        # Combine the '.snr' and '.dat.' DataFrames into a single acoustic DataFrame, make the timestamp
        # the index, and return an ADVMData object
        acoustic_df = pd.concat([dat_df, snr_df, vel_df], axis=1)

        data_set_suffix = " (" + configuration_parameters['Instrument'] + ")"
        data_origin = datamanager.DataManager.create_data_origin(acoustic_df, data_set_path + data_set_suffix)

        data_manager = datamanager.DataManager(acoustic_df, data_origin)

        return cls(data_manager, configuration_parameters)


class SL3GADVMData(ADVMData):

    def _calc_cell_range(self):
        """Calculate range of cells along a single beam.

        :return: Range of cells along a single beam
        """
        acoustic_data = self._data_manager.get_data()

        # Finds config dictionary definitions and determines cell sizes
        blanking_distance = self._configuration_parameters['Blanking Distance']
        cell_size = self._configuration_parameters['Cell Size']
        number_of_cells = self._configuration_parameters['Number of Cells']

        # 3G is blanking distance + 1.5* cell size
        first_cell_midpoint = blanking_distance + 1.5*cell_size
        last_cell_mid_point = first_cell_midpoint + (number_of_cells - 1)*cell_size

        # defines linspace
        cell_range = np.linspace(first_cell_midpoint, last_cell_mid_point, num=number_of_cells)
        cell_range = np.tile(cell_range, (acoustic_data.shape[0], 1))

        col_names = ['R{:03}'.format(cell) for cell in range(1, number_of_cells + 1)]
        cell_range_df = pd.DataFrame(data=cell_range, index=acoustic_data.index, columns=col_names)

        return cell_range_df

    @staticmethod
    def _read_mat_sample_date_time(mat_file):
        """Reads the sample date/time from the MAT file and converts to Pandas DateTimeIndex

        :param mat_file: Opened MAT file containing SL3G data
        :return:
        """

        # Read sample time in as Serial Number (in microseconds from Jan 1st, 2000) and give it a timestamp
        sample_time = mat_file['FlowData_SampleTime']
        sample_time = sample_time.flatten()
        sample_times = [datetime(2000, 1, 1, 0, 0) + timedelta(microseconds=x) for x in sample_time]

        # Set index as the timestamp
        datetime_index = pd.to_datetime(sample_times)

        return datetime_index

    @staticmethod
    def _read_mat_param(mat_file):
        """
        Read the Sontek Mat file into a configuration dictionary

        :param mat_file: Opened MAT file containing SL3G data
        :return: Dictionary containing specific configuration parameters
        """

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

        # Read Blanking Distance in as centimeters, return meters
        blanking_dist = mat_file['System_IqSetup'][0, 0]['advancedSetup']['SLblankingDistance']
        blanking_dist = blanking_dist / 1000
        config_dict['Blanking Distance'] = float(blanking_dist[0])

        # Read Cell Size in as centimeters, return meters
        cell_size = mat_file['System_IqSetup'][0, 0]['advancedSetup']['SLcellSize']
        cell_size = cell_size / 1000
        config_dict['Cell Size'] = float(cell_size[0])

        # Number of Cells
        number_cells = mat_file['System_IqSetup'][0, 0]['advancedSetup']['SLcellCount']
        config_dict['Number of Cells'] = int(number_cells[0])

        # Return created configuration dictionary
        return config_dict

    @staticmethod
    def _read_mat_dat(mat_file):
        """
        Read the '.mat' file and create a DataFrame for flow parameters

        :param mat_file: Opened MAT file containing SL3G data
        :return: Timestamp formatted DataFrame containing flow parameters
        """

        # Read in temperature
        temperature = mat_file['FlowData_Temp']
        temperature = temperature.flatten()

        # Read in vertical beam distance from Depth
        v_beam = mat_file['FlowData_Depth']
        v_beam = v_beam.flatten()

        # Create a DataFrame from variables above with 'index=timestamp'
        dat_df = pd.DataFrame({'Temp': temperature, 'Vbeam': v_beam})

        return dat_df

    @staticmethod
    def _read_mat_cell_amp(mat_file):
        """
        Read in .mat file and create a DataFrame with both beam's SNR data

        :param mat_file: Opened MAT file containing SL3G data
        :return: DataFrame for both beams SNR data
        """

        # Beam AMP data
        beam_one_amp = mat_file['Profile_0_Amp']
        beam_two_amp = mat_file['Profile_1_Amp']

        assert beam_one_amp.shape == beam_two_amp.shape

        amp_data = np.hstack((beam_one_amp, beam_two_amp))

        number_of_cells = beam_one_amp.shape[1]
        column_names = ['Cell{0:02}Amp{1:1}'.format(cell, beam) for beam in [1, 2]
                        for cell in range(1, number_of_cells+1)]

        # Create DataFrames for both beams
        amp_df = pd.DataFrame(data=amp_data, columns=column_names)

        return amp_df

    @staticmethod
    def _read_mat_cell_vel(mat_file):
        """Read the cell velocity from an SL3G MAT file

        :param mat_file: Opened MAT file containing SL3G data
        :return:
        """

        beam_1_velocity_key = 'Profile_0_Vel'
        beam_2_velocity_key = 'Profile_1_Vel'

        # transform the beam velocity to X, Y velocity
        beam_velocity = np.stack((mat_file[beam_1_velocity_key], mat_file[beam_2_velocity_key]), axis=1)
        vel_transform_matrix = [[1.183, -1.183], [0.552, 0.552]]  # TODO: Get more precise transform matrix
        xy_velocity = np.dot(vel_transform_matrix, beam_velocity)

        # convert the velocity units to m/s
        beam_1_vel_units = mat_file['Data_Units'][beam_1_velocity_key][0][0][0]
        beam_2_vel_units = mat_file['Data_Units'][beam_2_velocity_key][0][0][0]

        assert beam_1_vel_units == beam_2_vel_units

        if beam_1_vel_units == 'mm/s':
            vel_conversion = 1 / 1000
        elif beam_1_vel_units == 'm/s':
            vel_conversion = 1
        else:
            vel_conversion = np.nan

        xy_velocity = vel_conversion * xy_velocity

        # put the velocity data into a DataFrame
        number_of_cells = xy_velocity.shape[2]
        velocity_data = np.hstack([xy_velocity[i, :, :] for i in range(xy_velocity.shape[0])])
        column_names = ['Cell{0:02}V{1}'.format(cell, direction) for direction in ['x', 'y']
                        for cell in range(1, number_of_cells + 1)]
        vel_df = pd.DataFrame(data=velocity_data, columns=column_names)

        return vel_df

    @classmethod
    def read_sl3g_mat(cls, mat_file_path):
        """
        Loads the Sontek Data into the ADVM Data class object, retrieves data, snr and paramaters
        with cell location

        :param mat_file_path: file path containing the .mat file for the sontek data
        :return: ADVMData object containing the Sontek Data
        """

        mat_file = scipy.io.loadmat(mat_file_path, struct_as_record=True, squeeze_me=False)

        # Read the Argonaut '.mat' files into DataFrame
        dat_df = cls._read_mat_dat(mat_file)
        amp_df = cls._read_mat_cell_amp(mat_file)
        vel_df = cls._read_mat_cell_vel(mat_file)
        config_dict = cls._read_mat_param(mat_file)

        # Read specific configuration values from '.mat' file
        configuration_parameters = ADVMConfigParam()
        configuration_parameters.update(config_dict)

        # Combine both DataFrames and add date time index
        acoustic_df = pd.concat([dat_df, amp_df, vel_df], axis=1)
        datetime_index = cls._read_mat_sample_date_time(mat_file)
        acoustic_df.set_index(datetime_index, inplace=True)

        # create a data manager from the acoustic data
        data_origin = datamanager.DataManager.create_data_origin(acoustic_df, mat_file_path + "(SL3G)")
        data_manager = datamanager.DataManager(acoustic_df, data_origin)

        return cls(data_manager, configuration_parameters)
