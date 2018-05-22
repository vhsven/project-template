import warnings

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class CsvImporter(BaseEstimator, TransformerMixin):
    """Transform a CSV document to a numpy matrix of data such that the data 
    is ready for use by decision tree classifiers. This implies that instances 
    with missing values are removed and that one-hot encoding is applied to 
    all non-numeric columns. The class column is processed with a label encoder.

    Parameters
    ----------
    encoding : string, 'utf-8' by default.
        The encoding used to decode the input file.

    sep : str, default ','
        Delimiter to use.

    dtype : Type name or dict of column -> type, default None
        Data type for data or columns. E.g. {'a': np.float64, 'b': np.int32}. 
        Use str or object together with suitable na_values settings to preserve 
        and not interpret dtype.

    na_values : scalar, str, list-like, or dict, default None
        Additional strings to recognize as NA/NaN. If dict passed, specific 
        per-column NA values. The following values are always interpreted 
        as NaN: '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', 
        '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 
        'null'.

    class_index : int, default -1 (i.e., the last column)
        Column index of the class attribute. This column will not be present
        in the transform output, but will be kept separately in the y attribute
        of this transformer. Multi output scenarios are not supported.

    missing_threshold : float (percentage), default 0.75
        Indicates the least amount of data that must remain after removing
        instances with missing values without raising a warning. If less remain,
        a warning will be raised.


    Attributes
    ----------
    y : numpy array, [n_samples]
        Data extracted from the CSV based on the given `class_index` and then encoded. This data
        is not returned by transform, but saved here instead.

    original_y : numpy array, [n_samples]
        Same as `y`, but before encoding.
    """
    def __init__(self, encoding='utf-8', sep=',', dtype=None, na_values=None, class_index=-1, missing_threshold=0.75):
        self.encoding = encoding
        self.sep = sep
        self.dtype = dtype
        self.na_values = na_values
        self.class_index = class_index
        self.missing_threshold = missing_threshold
        self.y = None
        self.original_y = None

    def fit(self, csv_file, y=None):
        """Extract data from the given CSV file.

        Parameters
        ----------
        csv_file : string
            File path to CSV file.

        Returns
        -------
        self
        """
        self.fit_transform(csv_file)
        return self

    def fit_transform(self, csv_file, y=None):
        """Extract data from the given CSV file and return it as a numpy matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        csv_file : string
            File path to CSV file.

        Returns
        -------
        X : numpy matrix, [n_samples, n_features]
            Extracted data.
        """

        return self.transform(csv_file)

    def transform(self, csv_file):
        """Extract data from the given CSV file and return it as a numpy matrix.

        Parameters
        ----------
        csv_file : string
            File path to CSV file.

        Returns
        -------
        X : numpy matrix, [n_samples, n_features]
            Extracted data.
        """
        # Assumption: class is last column
        df = pd.read_csv(csv_file, encoding=self.encoding, sep=self.sep, dtype=self.dtype, na_values=self.na_values)
        n_before = len(df)
        #TODO alternative: Imputation http://scikit-learn.org/dev/modules/impute.html (assumes Guassian)
        df.dropna(inplace=True)
        n_after = len(df)
        percent_remaining = n_after / n_before
        percent_lost = 100 * (1 - percent_remaining)
        if percent_remaining < self.missing_threshold:
            warnings.warn("Lost {:.0f}% of instances by removing missing values: {} -> {}".format(percent_lost, n_before, n_after))
        self.original_y = df.iloc[:, self.class_index].values
        self.y = LabelEncoder().fit_transform(self.original_y)
        class_column_name = df.columns[self.class_index]
        df.drop(class_column_name, axis=1, inplace=True)
        
        # apply one hot encoding to non-numeric columns
        df2 = pd.get_dummies(df) # like CategoricalEncoder
        return df2.values

    # def inverse_transform(self, X): #TODO needed?
    #     raise NotImplementedError("Cannot convert back from data to CSV file.")

    def fit_transform_both(self, csv_file):
        """Extract data from the given CSV file and return it as a numpy matrix.
        Also returns the encoded class values at the same time.

        Parameters
        ----------
        csv_file : string
            File path to CSV file.

        Returns
        -------
        X : numpy matrix, [n_samples, n_features]
            Extracted data.
        
        y : numpy array, [n_samples]
            Data extracted from the CSV based on the given `class_index` and then encoded. 
        """
        X = self.fit_transform(csv_file)
        return X, self.y
