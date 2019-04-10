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
          "Iris-setosa": 0,
          "Iris-versicolor": 1,
          "Iris-virginica": 0
        }

        self.criteria = [
            'accuracy',
            'precision',
            'recall',
            'fmeasure',
            'specificity'
        ]

        self.results = []  # placeholder for the results.

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

    def accuracy(self, metrics):
        """Calculate the accuracy.

        metrics (dict) has all needed metrics (tn, tp, fn, tp)
        """
        sum = metrics['tp'] + metrics['tn'] + metrics['fp'] + metrics['fn']
        if (sum == 0):
            return 0

        value = (metrics['tp'] + metrics['tn']) / sum
        return value

    def precision(self, metrics):
        """Calculate the precision.

        metrics (dict) has all needed metrics (tn, tp, fn, tp)
        """
        sum = metrics['tp'] + metrics['fp']
        if (sum == 0):
            return 0

        value = metrics['tp'] / sum
        return value

    def recall(self, metrics):
        """Calculate the recall.

        metrics (dict) has all needed metrics (tn, tp, fn, tp)
        """
        sum = metrics['tp'] + metrics['fn']

        if (sum == 0):
            return 0

        value = (metrics['tp'] + metrics['fn']) / sum
        return value

    def fmeasure(self, metrics):
        """Calculate the fmeasure.

        metrics (dict) has all needed metrics (tn, tp, fn, tp)
        """
        sum = self.precision(metrics) + self.recall(metrics)

        if (sum == 0):
            return 0

        value = (self.precision(metrics) * self.recall(metrics)) / (sum * 2)
        return value

    def sensitivity(self, metrics):
        """Calculate the specificity.

        metrics (dict) has all needed metrics (tn, tp, fn, tp)
        """
        sum = metrics['tp'] + metrics['fn']

        if (sum == 0):
            return 0

        value = metrics['tp'] / sum
        return value

    def specificity(self, metrics):
        """Calculate the specificity.

        metrics (dict) has all needed metrics (tn, tp, fn, tp)
        """
        sum = metrics['tn'] + metrics['fp']

        if (sum == 0):
            return 0

        value = metrics['tn'] / sum
        return value

    def cross_validation(self):
        """Perform a cross validation test."""
        cross_validation = self.get_fold(self.folds, self.x, self.t)
        fold = next(cross_validation)
        fold_index = 0

        while (fold):
            try:
                # get the weight vector
                wT = self.train(fold['xtrain'], fold['ttrain'])

                # test the model using the weights.
                patternScores = self.test(fold['xtest'], fold['ttest'], wT)

                # call the predict function
                determened_output = []
                for i in range(len(patternScores)):
                    current_score = self.predict(patternScores[i])
                    determened_output.append(current_score)

                fold_results = {}
                # evaluate
                for criterion in self.criteria:
                    fold_results[criterion] = self.evaluate(fold['ttest'],
                                                            determened_output,
                                                            criterion)

                # save the results fot later calculations
                self.results.append(fold_results)

                # Add to plot
                fold_index += 1
                self.add_to_plot(fold['ttest'], determened_output, fold_index)

                # Call next fold
                fold = next(cross_validation)
            except StopIteration:
                break

    def train(self, patterns, targets):
        """Find the weights."""
        wT = np.linalg.pinv(patterns).dot(targets)
        return wT

    def test(self, patterns, targets, w):
        """Calculate the clasifier output for every pattern."""
        results = []
        for i in range(len(patterns)):
            yi = patterns[i].dot(w)
            results.append(yi)

        return results

    def predict(self, value):
        """Decide at what class a pattern belong to."""
        if value < 0:
            return 0

        return 1

    def evaluate(self, targets, predictions, criterion):
        """Evaluate the means of the basic metrics.

        targets (array): the real targets.
        predictions (array): the array with the claifier output.
        criterion (string): The criterion to be used.
        """
        if not any(list(map(lambda x: criterion == x, self.criteria))):
            raise Exception()

        tn = tp = fn = fp = 0

        for i in range(len(targets)):
            tn += (targets[i] == predictions[i]) and targets[i] == 0
            tp += (targets[i] == predictions[i]) and targets[i] == 1
            fn += (targets[i] != predictions[i]) and targets[i] == 1
            fp += (targets[i] != predictions[i]) and targets[i] == 0

        metrics = {
            'tn': tn,
            'tp': tp,
            'fn': fn,
            'fp': fp
        }

        calculated_values = {
            'accuracy': self.accuracy(metrics),
            'precision': self.precision(metrics),
            'recall': self.recall(metrics),
            'fmeasure': self.fmeasure(metrics),
            'sensitivity': self.sensitivity(metrics),
            'specificity': self.specificity(metrics),
        }

        return calculated_values[criterion]

    def add_to_plot(self, targets, predictions, position):
        """Add a subplot of the fold to the plot.

        targets: The real targets
        predictions: The calculated outputs
        position: The position of the plot inside the grid.
        """
        plt.subplot(3, 3, position)
        plt.plot(targets, 'bo')
        plt.plot(predictions, 'r.')
        plt.title('fold ' + str(position + 1))

    def show_plot(self):
        """Show the plot."""
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    a = LabSolver()
    a.cross_validation()
    a.show_plot()
    print ("done!")
