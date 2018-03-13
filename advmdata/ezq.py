import numpy as np
import pandas as pd
import linecache

from linearmodel.linearmodel import datamanager
from advmdata.advmdata.core import ADVMData, ADVMConfigParam


def combine_time_index(initial_df, wanted_index_df):

    initial_df_original_index = initial_df.index
    initial_df = initial_df.append(pd.DataFrame(index=wanted_index_df.index))
    initial_df.sort_index(inplace=True)
    initial_df.interpolate(inplace=True)
    initial_df.drop(labels=initial_df_original_index, axis=0, inplace=True)

    return initial_df


class EzqADVMData(ADVMData):

    def _calc_cell_range(self):

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
    def _read_param_from_hdr(data_file_path):

        hdr_file_path = data_file_path + '.hdr'
        config_dict = {}

        line = linecache.getline(hdr_file_path, 12).strip()
        line = ''.join(line.split())
        line = line.split("Cellsize")[-1:]
        cell_size = line[0]
        cell_size = cell_size[:-2]
        config_dict['Cell Size'] = float(cell_size)/100

        line = linecache.getline(hdr_file_path, 13).strip()
        line = ''.join(line.split())
        line = line.split("Blankingdistance")[-1:]
        blanking_distance = line[0]
        blanking_distance = blanking_distance[:-1]
        config_dict['Blanking Distance'] = float(blanking_distance)

        line = linecache.getline(hdr_file_path, 18).strip()
        line = ''.join(line.split())
        line = line.split("Diagnostics-Numberofbeams")[-1:]
        number_of_beams = line[0]
        config_dict['Number of Beams'] = int(number_of_beams)

        line = linecache.getline(hdr_file_path, 87).strip()
        line = ''.join(line.split())
        line = line.split("Headfrequency")[-1:]
        frequency = line[0]
        frequency = frequency[:-3]
        config_dict['Frequency'] = float(frequency)/1000

        data_file_path = data_file_path + '.ra1'
        line = linecache.getline(data_file_path, 1).split("   ")
        number_of_cells = len(line) - 6
        config_dict['Number of Cells'] = number_of_cells

        config_dict['Beam Orientation'] = 'Horizontal'
        config_dict['Slant Angle'] = float(25.0)
        config_dict['Instrument'] = 'AQD'
        config_dict['Effective Transducer Diameter'] = float(0.01395)

        return config_dict

    @staticmethod
    def _read_amp_from_ra(data_file_path, number_of_cells, beam_number):
        amp_column_names = ['Cell{:02}Amp{:1}'.format(cell, beam_number) for cell in range(1, number_of_cells + 1)]
        amp_file_path = data_file_path + '.ra' + str(beam_number)
        amp_df = pd.read_table(amp_file_path, header=None, sep='\s+')
        amp_df = amp_df.drop(amp_df.columns[0:11], axis=1)
        amp_df.columns = amp_column_names

        return amp_df

    @classmethod
    def _read_backscatter(cls, data_file_path, config_dict):

        datetime_index = cls._read_time_stamp(data_file_path)
        a_list = []
        for beam_number in range(1, config_dict['Number of Beams'] + 1):
            a_df = cls._read_amp_from_ra(data_file_path, config_dict['Number of Cells'], beam_number)
            a_list.append(a_df)

        a_list = pd.concat(a_list, axis=1)
        a_list = a_list.set_index(datetime_index)

        return a_list

    @staticmethod
    def _read_time_stamp(data_file_path):

        data_file_path = data_file_path + '.ra1'
        time_df = pd.read_table(data_file_path, header=None, sep='\s+')
        time_df = time_df.drop(time_df.columns[6:], axis=1)

        date_time_columns = ['Month', 'Day', 'Year', 'Hour', 'Minute', 'Second']
        time_df.columns = date_time_columns

        datetime_index = pd.to_datetime(time_df)

        return datetime_index

    @staticmethod
    def _read_sens_file(data_file_path):

        data_file_path = data_file_path + '.sen'
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
    def _read_dat_file(data_file_path):

        data_file_path = data_file_path + '.dat'
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
    def create_df(cls, data_file_path):

        v_beam_df = cls._read_sens_file(data_file_path)
        temp_df = cls._read_dat_file(data_file_path)
        config_dict = cls._read_param_from_hdr(data_file_path)

        configuration_parameters = ADVMConfigParam()
        configuration_parameters.update(config_dict)

        amp_df = cls._read_backscatter(data_file_path, config_dict)

        data_df = pd.concat([v_beam_df, temp_df], axis=1)

        data_df = combine_time_index(data_df, amp_df)

        advm_data_df = pd.concat([data_df, amp_df], axis=1)
        advm_data_origin = datamanager.DataManager.create_data_origin(advm_data_df, data_file_path + "(EzQ)")

        advm_data_manager = datamanager.DataManager(advm_data_df, advm_data_origin)

        return cls(advm_data_manager, configuration_parameters)





