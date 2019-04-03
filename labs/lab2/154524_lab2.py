"""The  code for the lab1."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

try:
    # if runs on student's local machine, try get the local file.
    from local_configuration import LocalConfiguration
    configuration = LocalConfiguration()
    data = configuration.read_data()
except:
    # Else get file from an online resource.
    url_data = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    # 2
    data = pd.read_csv(url_data, header=None).values


def evaluate(t, predict, criterion):
    """Evaluate the standard metrics."""
    tn = 0.
    fn = 0.
    tp = 0.
    fp = 0.

    if not len(predict) == len(t):
        raise ValueError(" List Arguments do not have the same length ")

    for i in range(len(t)):
        tn += predict[i] == t[i] == 0
        fn += predict[i] == 0 and t[i] == 1
        tp += predict[i] == t[i] == 1
        fp += predict[i] == 1 and t[i] == 0

    select = {
        'accuracy': ((tp + tn) / (tp + tn + fp + fn)),
        'precision': (tp / (tp + fp)),
        'recall': (tp / (tp + fn)),
        'sensitivity': (tp / (tp + fn)),
        'specificity': (tn / (tn + fp))
    }

    # :)
    select.update({'fmeasure': ((select['precision'] * select['recall']) / (select['precision'] + select['recall']) / 2)})

    return select[criterion]


# data
NumberOfPatterns, NumberOfAttributes = data.shape

map_dict = {
  "Iris-setosa": 0,
  "Iris-versicolor": 1,
  "Iris-virginica": 0
}

x = data[:, :4]

t = np.zeros(NumberOfPatterns)

for pattern in range(NumberOfPatterns):
    t[pattern] = map_dict[data[pattern][4]]

# Add 1 at the patterns.
aces = np.ones([NumberOfPatterns, 1])

labels = ['accuracy', 'sensitivity', 'recall', 'precision', 'fmeasure', 'specificity']
all_values = []

for k in range(9):
    new_x = np.hstack((x, aces))
    xtrain, xtest, ttrain, ttest = train_test_split(new_x, t, test_size=0.1)

    ttrain1 = list(map(lambda x: 1 if x == 1 else -1, ttrain))
    ttest1 = list(map(lambda x: 1 if x == 1 else -1, ttest))

    # convert xtrain to numpy.float64 type
    # else numpy.linalg.pinv() will raise exceptio
    xtrain = list(map(lambda x: np.float64(x), xtrain))
    wT = np.linalg.pinv(xtrain).dot(ttrain1)

    # calculate the y
    y = new_x.dot(wT)

    # clasifier output
    ytest = xtest.dot(wT)

    predict_test = list(map(lambda x: 0 if x < 0 else 1, ytest))

    values = []

    for i in range(len(labels)):
        try:
            val = evaluate(ttest, predict_test, labels[i])
        except:
            val = 0
            values.append(val)

    plt.subplot(3, 3, k+1)
    plt.plot(ttest, 'bo')
    plt.plot(predict_test, 'ro')
    plt.title('fold ' + str(k+1))

    all_values.append(values)

# calculate means
for i in range(len(labels)):
    s = 0
    for j in range(len(all_values)):
        s += all_values[j][i]

    # s = sum(all_values[:,i:(i+1)])
    mo = s / len(all_values)
    print(labels[i], "\t mean: \t", mo)

plt.tight_layout()
plt.show()
