# Authors:
# Bryce Walter Parker <brycewalterparker at gmail dot com>
# Jackson Hoffart <jackson dot hoffart at gmail dot com>

# License: GPL-3.0

# include this when deploying!
# ! pip freeze > requirements.txt

''' import some cheeky modules '''
# standard modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

%matplotlib inline

''' sci-kit learn handwritten digit recognition tutorial'''
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
