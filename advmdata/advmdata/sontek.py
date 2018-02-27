import linecache
import numpy as np
import os
import pandas as pd
import scipy
from datetime import datetime
from datetime import timedelta
from scipy import io

from advmdata.advmdata.core import ADVMData, ADVMConfigParam
from linearmodel.linearmodel import datamanager

class SonTek3g(ADVMData):

    def _calc_cell_range(self):
        """Calculate range of cells along a single beam.

        :return: Range of cells along a single beam
        """
        acoustic_data = self._data_manager.get_data()

        blanking_distance = self._configuration_paramters['Blanking Distance']
        cell_size = self._configuration_parameters['Cell Size']
        number_of_cells = self._configuration_parameters['Number of Cells']

        first_cell_midpoint = blanking_distance + cell_size / 2
        last_cell_mid_point = first_cell_midpoint + (number_of_cells - 1) * cell_size

        cell_range = np.linspace(first_cell_midpoint, last_cell_mid_point, num = number_of_cells)
        cell_range = np.tile(cell_range, (acoustic_data.shape[0], 1))

        col_names = ['R{:03}'.format(cell) for cell in range(1, number_of_cells + 1)]

        cell_range_df = pd.DataFrame(data=cell_range, index=acoustic_data.index, columns =col_names)
        return cell_range_df

    @staticmethod
    def _read_mat_file_param(sontek_filepath):
        mat_file = scipy.io.loadmat(sontek_filepath, struct_as_record=True, squeeze_me=False)
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

        # Default Slant Angle
        config_dict['Slant Angle'] = 25

        # Default Number of Beams
        config_dict['Number of Beams'] = int(2)

        # Read Blanking Distance in as CM, return M
        blanking_dist = mat_file['System_IqSetup'][0, 0]['advancedSetup']['SLblankingDistance']
        if blanking_dist > 100:
            blanking_dist = blanking_dist / 100
        config_dict['Blanking distance'] = float(blanking_dist[0])

        # Read Cell Size in as CM, return M
        cell_size = mat_file['System_IqSetup'][0, 0]['advancedSetup']['SLcellSize']
        if cell_size > 100:
            cell_size = cell_size/100
        config_dict['Cell Size'] = float(cell_size[0])

        number_cells = mat_file['System_IqSetup'][0, 0]['advancedSetup']['SLcellCount']
        config_dict['Number of Cells'] = float(number_cells[0])

        return config_dict

    @staticmethod
    def _read_mat_file_dat(sontek_filepath):
        mat_file = scipy.io.loadmat(sontek_filepath, struct_as_record=True, squeeze_me=False)

        sample_time = mat_file['FlowData_SampleTime']
        sample_time = sample_time.flatten()
        sample_times = [datetime(2000,1,1,0,0) + timedelta(microseconds=x) for x in sample_time]

        datetime_index = pd.to_datetime(sample_times)

        sample_number = mat_file['FlowData_SampleNumber']
        sample_number = sample_number.flatten()

        temperature = mat_file['FlowData_Temp']
        temperature = temperature.flatten()

        v_beam = mat_file['FlowData_Depth']
        v_beam = v_beam.flatten()

        dat_df = pd.DataFrame({'Temp': temperature, 'Vertical Beam': v_beam}, index=sample_number)
        dat_df.set_index(datetime_index, inplace=True)

        return dat_df

    @staticmethod
    def _read_sontek_snr(sontek_filepath):

        mat_file = scipy.io.loadmat(sontek_filepath, struct_as_record=True, squeeze_me=False)

        intensity_factor = np.NaN
        beam_one_amp = mat_file['Profile_0_Amp']
        beam_snr = mat_file['FlowData_SNR']
        beam_one_snr = intensity_factor * (beam_one_amp - beam_snr[:, 0, None])

        beam_two_amp = mat_file['Profile_1_Amp']
        beam_snr = mat_file['FlowData_SNR']
        beam_two_snr = intensity_factor * (beam_two_amp - beam_snr[:, 1, None])

        snr_df1 = pd.DataFrame(beam_one_snr)
        snr_df2 = pd.DataFrame(beam_two_snr)
        snr_df = pd.concat([snr_df1, snr_df2], axis=1)

        return snr_df

    @staticmethod
    def _read_argonaut_to_df(cls, data_directory, filename):

        dataset_path = os.path.join(data_directory, filename)

        sontek_mat_file = dataset_path + ".mat"
        dat_df = cls.__read_mat_file_dat(sontek_mat_file)
        snr_df = cls.__read_sontek_snr(sontek_mat_file)

        config_dict = cls._read_mat_file_param(sontek_mat_file)
        configuration_parameters = ADVMConfigParam()
        configuration_parameters.update(config_dict)

        acoustic_df = pd.concat([dat_df, snr_df], axis=1)

        data_origin = datamanager.DataManager.create_data_origin(acoustic_df, dataset_path + "(Sontek")

        data_manager = datamanager.DataManager(acoustic_df, data_origin)

        return cls(data_manager, configuration_parameters)








