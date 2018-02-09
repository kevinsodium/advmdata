import abc
import copy
import re

import numpy as np

from linearmodel import datamanager


class ADVMError(Exception):
    """Base class for exceptions in the advmdata package"""
    pass


class ADVMDataIncompatibleError(ADVMError):
    """An error if ADVMData instances are incompatible"""
    pass


class ADVMParam(abc.ABC):
    """Base class for ADVM parameter classes"""

    @abc.abstractmethod
    def __init__(self, param_dict):
        self._dict = param_dict

    def __deepcopy__(self, memo):
        """
        Provide method for copy.deepcopy().

        :param memo:
        :return:
        """

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v, in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __getitem__(self, key):
        """Return the requested parameter value.

        :param key: Parameter key
        :return: Parameter corresponding to the given key
        """
        return self._dict[key]

    def __repr__(self):
        return self._dict.__repr__()

    def __setitem__(self, key, value):
        """Set the requested parameter value.

        :param key: Parameter key
        :param value: Value to be stored
        :return: Nothing
        """

        self._check_value(key, value)
        self._dict[key] = value

    def __str__(self):
        return self._dict.__str__()

    @abc.abstractmethod
    def _check_value(self, key, value):
        raise NotImplementedError

    def _check_key(self, key):
        """Check if the provided key exists in the _dict. Raise an exception if not.

        :param key: Parameter key to check
        :return: Nothing
        """

        if key not in self._dict.keys():
            raise KeyError(key)
        return

    def get_dict(self):
        """Return a dictionary containing the processing parameters.

        :return: Dictionary containing the processing parameters
        """

        return copy.deepcopy(self._dict)

    def items(self):
        """Return a set-like object providing a view on the contained parameters.

        :return: Set-like object providing a view on the contained parameters
        """

        return self._dict.items()

    def keys(self):
        """Return the parameter keys.

        :return: A set-like object providing a view on the parameter keys.
        """

        return self._dict.keys()

    def update(self, update_values):
        """Update the  parameters.

        :param update_values: Item containing key/value processing parameters
        :return: Nothing
        """

        for key, value in update_values.items():
            self._check_value(key, value)
            self._dict[key] = value


class ADVMConfigParam(ADVMParam):
    """Stores ADVM Configuration parameters."""

    def __init__(self):
        """
        """

        # the valid for accessing information in the configuration parameters
        valid_keys = ['Frequency', 'Effective Transducer Diameter', 'Beam Orientation', 'Slant Angle',
                      'Blanking Distance', 'Cell Size', 'Number of Cells', 'Number of Beams']

        # initial values for the configuration parameters
        init_values = np.tile(np.nan, (len(valid_keys),))

        config_dict = dict(zip(valid_keys, init_values))

        super().__init__(config_dict)

    def is_compatible(self, other):
        """Checks compatibility of ADVMConfigParam instances

        :param other: Other instance of ADVMConfigParam
        :return:
        """

        keys_to_check = ['Frequency', 'Slant Angle', 'Blanking Distance', 'Cell Size', 'Number of Cells']

        compatible_configs = True

        for key in keys_to_check:
            if not self[key] == other[key]:
                compatible_configs = False
                break

        return compatible_configs

    def _check_value(self, key, value):
        """Check if the provided value is valid for the given key. Raise an exception if not.

        :param key: Keyword for configuration item
        :param value: Value for corresponding key
        :return: Nothing
        """

        self._check_key(key)

        other_keys = ['Frequency', 'Effective Transducer Diameter', 'Slant Angle', 'Blanking Distance', 'Cell Size']

        if key == "Beam Orientation" and (value == "Horizontal" or value == "Vertical"):
            return
        elif key == "Number of Cells" and (1 <= value and isinstance(value, int)):
            return
        elif key == "Number of Beams" and (0 <= value and isinstance(value, int)):
            return
        elif key in other_keys and 0 <= value and isinstance(value, (int, float)):
            return
        else:
            raise ValueError(value, key)


class ADVMData:

    # regex string to find ADVM data columns
    _advm_columns_regex = r'^(Temp|Vbeam|Cell\d{2}(Amp|SNR)\d{1})$'

    def __init__(self, data_manager, configuration_parameters):

        self._configuration_parameters = copy.deepcopy(configuration_parameters)

        # get the ADVM data only from the passed data manager
        acoustic_data = data_manager.get_data()
        acoustic_df = acoustic_data.filter(regex=self._advm_columns_regex)

        data_origin = data_manager.get_origin()
        data_variable_list = list(data_origin['variable'])
        acoustic_variable_idx = [i for i, item in enumerate(data_variable_list)
                                 if re.search(self._advm_columns_regex, item)]
        acoustic_data_origin = data_origin.ix[acoustic_variable_idx]

        self._data_manager = datamanager.DataManager(acoustic_df, acoustic_data_origin)

    def __deepcopy__(self, memo):
        """
        Provide method for copy.deepcopy().

        :param memo:
        :return:
        """

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v, in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    @abc.abstractmethod
    def _calc_cell_range(self):
        pass

    def add_data(self, other, keep_curr_obs=None):
        """Adds other ADVMData instance to self.

        Throws exception if other ADVMData object is incompatible with self. An exception will be raised if
        keep_curr_obs=None and concurrent observations exist for variables.

        :param other: ADVMData object to be added
        :type other: ADVMData
        :param keep_curr_obs: {None, True, False} Flag to indicate whether or not to keep current observations.
        :return: Merged ADVMData object
        """

        # test compatibility of other data set
        if not self._configuration_parameters.is_compatible(other.get_configuration_parameters()) and \
                isinstance(other, type(self)):

            raise ADVMDataIncompatibleError("ADVM data sets are incompatible")

        other_data = other.get_data()
        other_origin = other.get_origin()

        other_data_manager = datamanager.DataManager(other_data, other_origin)

        combined_data_manager = self._data_manager.add_data_manager(other_data_manager, keep_curr_obs=keep_curr_obs)

        return type(self)(combined_data_manager, self._configuration_parameters)

    def get_cell_range(self):
        """Get a DataFrame containing the range of cells along a single beam.

        :return: Range of cells along a single beam
        """

        return self._calc_cell_range()

    def get_configuration_parameters(self):
        """

        :return:
        """

        return copy.deepcopy(self._configuration_parameters)

    def get_data(self):
        """

        :return:
        """

        return self._data_manager.get_data()

    def get_data_manager(self):
        """

        :return:
        """

        return copy.deepcopy(self._data_manager)

    def get_origin(self):
        """

        :return:
        """

        return self._data_manager.get_origin()

    def get_variable(self, variable_name):
        """

        :param variable_name:
        :return:
        """

        return self._data_manager.get_variable(variable_name)

    def get_variable_names(self):
        """

        :return:
        """

        return self._data_manager.get_variable_names()

    def get_variable_observation(self, variable_name, time, time_window_width=0, match_method='nearest'):
        """

        :param variable_name:
        :param time:
        :param time_window_width:
        :param match_method:
        :return:
        """

        return self._data_manager.get_variable_observation(variable_name, time, time_window_width, match_method)

    def get_variable_origin(self, variable_name):
        """

        :param variable_name:
        :return:
        """

        data_origin = self.get_origin()
        grouped = data_origin.groupby('variable')
        variable_group = grouped.get_group(variable_name)
        variable_origin = list(variable_group['origin'])

        return variable_origin

    @classmethod
    def read_tab_delimited_data(cls, file_path, configuration_parameters):
        """Create an ADVMData object from a tab-delimited text file that contains raw acoustic variables.

        ADVM configuration parameters must be provided as an argument.

        :param file_path: Path to tab-delimited file
        :type file_path: str
        :param configuration_parameters: ADVMConfigParam containing necessary ADVM configuration parameters
        :type configuration_parameters: ADVMConfigParam

        :return: ADVMData object
        """

        data_manager = datamanager.DataManager.read_tab_delimited_data(file_path)

        return cls(data_manager, configuration_parameters)
