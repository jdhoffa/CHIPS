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

# scikit-learn support vector machines (classifier) and performance metrics
from sklearn import svm, metrics

# Cropped FEN diagrams, 240x240 pixels, 57600 pixels in total.
# Chess board is 8x8 spaces, at 30x30 pixels per space.
# Goal is to segment the diagram into 8x8=64 spaces (preserving their order).
# For each segmented space, we want to classify it by an id between (0,6)
# (0,1,2,3,4,5,6) = (king, queen, rook, bishop, knight, pawn, empty)
# For now, we will do so by flattening each image (after segmentation) and running a support vector machine based classifier on the dataset (as per scikit-learns image classification tutorial).

# Note: This is a toy problem, and should be very easy as each image is perfectly identical. Once this runs smoothly, we will tackle the more difficult problem of analyzing imperfect data and finally physical images of real chess boards.

# Load training data

train1 = matplotlib.pyplot.imread('../fen_diagrams/train1.png')

for filename in os.listdir('../fen_diagrams/'):
    if filename.endswith(".png"):
        a = matplotlib.pyplot.imread('../fen_diagrams/'+filename)
        continue
    else:
        continue
