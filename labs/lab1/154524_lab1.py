import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

url_data = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

#data = pandas.read_csv(url_data, header='None').values


#2
data = pd.read_csv(url_data, header=None).values


# 3

NumberOfPatterns, NumberOfAttributes = data.shape

map_dict = {
  "Iris-setosa": 0,
  "Iris-versicolor": 1,
  "Iris-virginica": 0
}

# 4
x = data[:,:4]

t = np.zeros(NumberOfPatterns)

for pattern in range(NumberOfPatterns):
  t[pattern] = map_dict[data[pattern][4]]


# 5
for k in range(9):
  xtrain, xtest, ttrain, ttest = train_test_split(x, t, test_size=0.1)
  plt.subplot(3, 3, k+1)
  plt.plot(xtrain[:,0], xtrain[:,2], 'bo')
  plt.plot(xtest[:,0], xtest[:,2], 'ro')
  plt.title('plot ' + str(k))

plt.tight_layout()
plt.show()
