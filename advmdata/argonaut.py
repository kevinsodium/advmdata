import linecache
import numpy as np
import os
import pandas as pd

from advmdata.core import ADVMData, ADVMConfigParam
from linearmodel import datamanager


class ArgonautADVMData(ADVMData):
    """Instrument-specific data for the SonTek Argonaut SL (second generation SL)"""

    def _calc_cell_range(self):
        """Calculate range of cells along a single beam.

        :return: Range of cells along a single beam
        """

        acoustic_data = self._data_manager.get_data()

        blanking_distance = self._configuration_parameters['Blanking Distance']
        cell_size = self._configuration_parameters['Cell Size']
        number_of_cells = self._configuration_parameters['Number of Cells']

        first_cell_mid_point = blanking_distance + cell_size / 2
        last_cell_mid_point = first_cell_mid_point + (number_of_cells - 1) * cell_size

        cell_range = np.linspace(first_cell_mid_point, last_cell_mid_point, num=number_of_cells)

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

        # Read specific configuration values from the Argonaut '.ctl' file into a dictionary.
        # The fixed formatting of the '.ctl' file is leveraged to extract values from foreknown file lines.
        config_dict = {}
        line = linecache.getline(arg_ctl_filepath, 10).strip()
        arg_type = line.split("ArgType ------------------- ")[-1:]

        if arg_type == "SL":
            config_dict['Beam Orientation'] = "Horizontal"
        else:
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
        relevant_columns = ['Temp', 'Vbeam']
        dat_df = dat_df.filter(regex=r'(' + '|'.join(relevant_columns) + r')$')

        dat_df = dat_df.apply(pd.to_numeric, args=('coerce', ))
        dat_df = dat_df.astype(np.float)

        return dat_df

    @staticmethod
    def _read_argonaut_snr_file(arg_snr_filepath):
        """
        Read the Argonaut '.dat' file into a DataFrame.

        :param arg_snr_filepath: Filepath containing the Argonaut '.dat' file
        :return: Timestamp formatted DataFrame containing '.snr' file contents
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
        snr_df = snr_df.filter(regex=r'(^Cell\d{2}(Amp|SNR)\d{1})$')

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
        snr_df = cls._read_argonaut_snr_file(arg_snr_file)

        # Read specific configuration values from the Argonaut '.ctl' file into a dictionary.
        arg_ctl_file = dataset_path + ".ctl"
        config_dict = cls._read_argonaut_ctl_file(arg_ctl_file)
        configuration_parameters = ADVMConfigParam()
        configuration_parameters.update(config_dict)

        # Combine the '.snr' and '.dat.' DataFrames into a single acoustic DataFrame, make the timestamp
        # the index, and return an ADVMData object
        acoustic_df = pd.concat([dat_df, snr_df], axis=1)
        data_origin = datamanager.DataManager.create_data_origin(acoustic_df, dataset_path + "(Arg)")

        data_manager = datamanager.DataManager(acoustic_df, data_origin)

        return cls(data_manager, configuration_parameters)
