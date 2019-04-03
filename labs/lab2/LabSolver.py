"""Solves a lab exersice."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class LabSolver(object):
    """Able to solve a machine learning lab."""

    def __init__(self):
        """Initialize the object."""
        self.folds = 9

        self.map_dict = {
          "Iris-setosa": -1,
          "Iris-versicolor": 1,
          "Iris-virginica": -1
        }

        self.data = None
        self.data = self.get_data()
        self.prepare_data()

    def read_data(self):
        """Read data using the local  configuration module."""
        try:
            # if runs on student's local machine, try get the local file.
            from local_configuration import LocalConfiguration
            configuration = LocalConfiguration()
            data = configuration.read_data()
        except:
            print ("Data not found, fetching")
            # Else get file from an online resource.
            url_data = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
            # 2
            data = pd.read_csv(url_data, header=None).values

        return data

    def get_data(self):
        """Return the data. Fetch the data if needed."""
        if self.data is None:
            self.data = self.read_data()

        return self.data

    def prepare_data(self):
        """Format the data."""
        data = self.get_data()
        NumberOfPatterns, NumberOfAttributes = data.shape

        patterns = data[:, :4]
        # Make patterns float type.
        patterns = list(map(lambda x: np.float64(x), patterns))

        targets = np.zeros(NumberOfPatterns)

        for pattern in range(NumberOfPatterns):
            targets[pattern] = self.map_dict[data[pattern][4]]

        # Add 1 at the patterns.
        aces = np.ones([NumberOfPatterns, 1])
        self.patterns = patterns
        self.x = np.hstack((patterns, aces))
        self.t = targets

    def get_fold(self, folds, x, t):
        """Slit the data and return a fold."""
        for k in range(folds):
            xtrain, xtest, ttrain, ttest = train_test_split(x, t,
                                                            test_size=0.1)
            fold = {
                'xtrain': xtrain,
                'xtest': xtest,
                'ttrain': ttrain,
                'ttest': ttest
            }

            yield fold

    def cross_validation(self):
        """Perform a cross validation test."""
        cross_validation = self.get_fold(self.folds, self.patterns, self.t)
        fold = next(cross_validation)

        while (fold):
            try:
                fold = next(cross_validation)
                self.train(fold['xtrain'], fold['ttrain'])
            except StopIteration:
                break

    def train(self, patterns, targets):
        """Find the weights."""
        wT = np.linalg.pinv(patterns).dot(targets)

        patterns = np.array(patterns)

        # calculate the y
        y = patterns.dot(wT)

    def evaluate(self):
        """Evaluate the means of the basic metrics."""
        pass


if __name__ == "__main__":
    a = LabSolver()
    a.cross_validation()
