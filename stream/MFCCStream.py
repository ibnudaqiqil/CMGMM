import os
import pandas as pd
import numpy as np

from skmultiflow.data.base_stream import Stream
from skmultiflow.data.data_stream import check_data_consistency


class MFCCStream(Stream):
    """ Creates a MFCC stream from a file source.

    For the moment only csv files are supported, but the goal is to support different formats,
    as long as there is a function that correctly reads, interprets, and returns
    a pandas' DataFrame or numpy.ndarray with the data.

    Parameters
    ----------
    filepath:
        Path to the data file

    target_idx: int, optional (default=-1)
        The column index from which the targets start.

    n_targets: int, optional (default=1)
        The number of targets.

    cat_features: list, optional (default=None)
        A list of indices corresponding to the location of categorical features.

    """
    _CLASSIFICATION = 'classification'
    _REGRESSION = 'regression'

    def __init__(self, filepath, nama_model="", additional_data=None, target_idx=-1, n_targets=1, cat_features=None, allow_nan=False, wave_column="mfcc", label_column="label"):
        super().__init__()

        self.filepath = filepath
        self.n_targets = n_targets
        self.target_idx = target_idx
        self.cat_features = cat_features
        self.cat_features_idx = [] if self.cat_features is None else self.cat_features
        self.allow_nan = allow_nan
        self.additional_data=additional_data
        self.X = None
        self.y = None
        self.task_type = None
        self.n_classes = 0
        self.filename = ''
        self.basename = ''

        # Automatically infer target_idx if not passed in multi-output problems
        if self.n_targets > 1 and self.target_idx == -1:
            self.target_idx = -self.n_targets
        self.wave_column = wave_column
        self.label_column = label_column
        self.nama_model =nama_model
        self.basename = os.path.basename(self.filepath)
        filename, extension = os.path.splitext(self.basename)
        if extension.lower() == '.csv':
            self.read_function = pd.read_csv
        elif extension.lower() == '.pickle':
            self.read_function = pd.read_pickle
        else:
            raise ValueError('Unsupported format: ', extension)
        self.filename = filename

        self._prepare_for_use()

    @property
    def target_idx(self):
        """
        Get the number of the column where Y begins.

        Returns
        -------
        int:
            The number of the column where Y begins.
        """
        return self._target_idx

    @target_idx.setter
    def target_idx(self, target_idx):
        """
        Sets the number of the column where Y begins.

        Parameters
        ----------
        target_idx: int
        """

        self._target_idx = target_idx

    @property
    def n_targets(self):
        """
         Get the number of targets.

        Returns
        -------
        int:
            The number of targets.
        """
        return self._n_targets

    @n_targets.setter
    def n_targets(self, n_targets):
        """
        Sets the number of targets.

        Parameters
        ----------
        n_targets: int
        """

        self._n_targets = n_targets

    @property
    def cat_features_idx(self):
        """
        Get the list of the categorical features index.

        Returns
        -------
        list:
            List of categorical features index.

        """
        return self._cat_features_idx

    @cat_features_idx.setter
    def cat_features_idx(self, cat_features_idx):
        """
        Sets the list of the categorical features index.

        Parameters
        ----------
        cat_features_idx:
            List of categorical features index.
        """

        self._cat_features_idx = cat_features_idx

    def _prepare_for_use(self):
        self.restart()
        self._load_data()

    def _load_data(self):
        """ Reads the data provided by the user and separates the features and targets.
        """
        #print("load data")
        try:

            raw_datax = self.read_function(self.filepath)
            #if (self.additional_data.empty()):
           # print(raw_datax.info())
            #print(self.additional_data.info())
            
            frames = [self.additional_data, raw_datax]
            raw_data = pd.concat(frames).reset_index(drop=True)
            #print(raw_data.shape)
        #check_data_consistency(raw_data, self.allow_nan)

            rows, cols = raw_data.shape
            #print(raw_data.shape)
            self.n_samples = rows
            
            labels = raw_data['label'].unique().tolist()
            mapping = dict( zip(labels,range(len(labels))) )
            raw_data.replace({'label': mapping},inplace=True)

         
            self.y = raw_data.label.to_numpy()
            self.target_names = "label"
            self.X = raw_data.mfcc.to_numpy()
            self.feature_names ="mfcc"
            #print(len(self.X))
            
            self.n_num_features = 1
            
            
            self.task_type = self._CLASSIFICATION
            self.n_classes = len(np.unique(self.y))
            #print( self.n_classes )
            self.target_values = self.get_target_values()
        except FileNotFoundError:
            raise FileNotFoundError("File {} does not exist.".format(self.filepath))
        pass

    def restart(self):
        """ Restarts the stream.

        It basically server the purpose of reinitializing the stream to
        its initial state.

        """
        self.sample_idx = 0
        self.current_sample_x = None
        self.current_sample_y = None

    def next_sample(self, batch_size):
        """ Returns next sample from the stream.

        If there is enough instances to supply at least batch_size samples, those
        are returned. If there aren't a tuple of (None, None) is returned.

        Parameters
        ----------
        batch_size: int (optional, default=1)
            The number of instances to return.

        Returns
        -------
        tuple or tuple list
            Returns the next batch_size instances.
            For general purposes the return can be treated as a numpy.ndarray.

        """
        #print(self.sample_idx)
        self.sample_idx += batch_size
        #print(batch_size)
        
        try:

            self.current_sample_x = self.X[self.sample_idx - batch_size:self.sample_idx]
            self.current_sample_y = self.y[self.sample_idx - batch_size:self.sample_idx]
            if self.n_targets < 2:
                self.current_sample_y = self.current_sample_y.flatten()

        except IndexError:
            print("xxxxxx")
            self.current_sample_x = None
            self.current_sample_y = None
        return self.current_sample_x, self.current_sample_y

    def has_more_samples(self):
        """ Checks if stream has more samples.

        Returns
        -------
        Boolean
            True if stream has more samples.

        """
        return (self.n_samples - self.sample_idx) > 0

    def n_remaining_samples(self):
        """ Returns the estimated number of remaining samples.

        Returns
        -------
        int
            Remaining number of samples.

        """
        return self.n_samples - self.sample_idx

    def get_all_samples(self):
        """
        returns all the samples in the stream.

        Returns
        -------
        X: pd.DataFrame
            The features' columns.
        y: pd.DataFrame
            The targets' columns.
        """
        return self.X, self.y

    def get_data_info(self):
        if self.task_type == self._CLASSIFICATION:
            return "{} - {} target(s), {} classes".format(self.nama_model, self.n_targets,
                                                          self.n_classes)
        elif self.task_type == self._REGRESSION:
            return "{} - {} target(s)".format(self.basename, self.n_targets)

    def get_target_values(self):
        if self.task_type == 'classification':
            if self.n_targets == 1:
                return np.unique(self.y).tolist()
            else:
                return [np.unique(self.y[:, i]).tolist() for i in range(self.n_targets)]
        elif self.task_type == self._REGRESSION:
            return [float] * self.n_targets

    def get_info(self):
        return 'MFCCStream(filename={}, target_idx={}, n_targets={}, cat_features={})' \
            .format("'" + self.basename + "'", self.target_idx, self.n_targets, self.cat_features)
