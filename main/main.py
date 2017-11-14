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
import os

# scikit-learn support vector machines (classifier) and performance metrics
from sklearn import svm, metrics

# Cropped FEN diagrams, 240x240 pixels, 57600 pixels in total.
# Chess board is 8x8 spaces, at 30x30 pixels per space.
# Goal is to segment the diagram into 8x8=64 spaces (preserving their order).
# For each segmented space, we want to classify it by an id between (0,6)
# (0,1,2,3,4,5,6) = (king, queen, rook, bishop, knight, pawn, empty)
# For now, we will do so by flattening each image (after segmentation) and running a support vector machine based classifier on the dataset (as per scikit-learns image classification tutorial).

# Note: This is a toy problem, and should be very easy as each image is perfectly identical. Once this runs smoothly, we will tackle the more difficult problem of analyzing imperfect data and finally physical images of real chess boards.

print ("---> We begin by loading in .PNGs of FEN diagrams (depicting the Scotch opening). Each diagram is 240x240 pixels, with 8x8 spaces (so 30x30 pixels per space).:")

# Note: This returns a Numpy array of size (240,240,4), 240x240 pixels with 4 colour elements (RGB and a transparency layer, I think?)

# Here, we will load all N training examples into a (N,240,240,4) numpy array:
a = []

for filename in os.listdir('../fen_diagrams/unaffected/'):
    if filename.endswith(".png"):
        a.append(matplotlib.pyplot.imread('../fen_diagrams/unaffected/'+filename))
        continue
    else:
        continue

dataset = np.stack(a)

# Plot a subset of the training examples

Nplots = int(raw_input("How many of the training examples would you like to see? (Maximum is {}): \n".format(len(a))))

f, axarr = plt.subplots(Nplots, sharex=True, figsize=(5,Nplots*7))

if Nplots != 1:
    for i in range(0,Nplots):
        axarr[i].imshow(dataset[i], cmap=plt.cm.gray_r, interpolation='nearest', aspect='auto')
        axarr[i].set_title('Training example: {}'.format((i+1)))
        axarr[i].axis('off')
else:
    axarr.imshow(dataset[0], cmap=plt.cm.gray_r, interpolation='nearest', aspect='auto')
    axarr.set_title('Training example: {}'.format((1)))
    axarr.axis('off')
plt.show()

raw_input("Press Enter to continue...")

print("We will now train the SVM classifier on the first training example:")
# Piece Key: 0,1,2,3,4,5,6 = empty, pawn, rook, knight, bishop, queen, king

plt.imshow(dataset[0], cmap=plt.cm.gray_r, interpolation='nearest', aspect='auto')
plt.title('Validation Set')
plt.axis('off')
plt.show()

raw_input("Press Enter to continue...")

print("The piece key is: [0,1,2,3,4,5,6] = [empty, pawn, rook, knight, bishop, queen, king].")
train1_validate=np.array((2,1,0,0,0,0,1,2, \
                          3,1,0,0,0,0,1,3, \
                          4,1,0,0,0,0,1,4, \
                          5,1,0,0,0,0,1,5, \
                          6,1,0,0,0,0,1,6, \
                          4,1,0,0,0,0,1,4, \
                          3,1,0,0,0,0,1,3, \
                          2,1,0,0,0,0,1,2))

print("The training values are:")
for i in range(0,len(train1_validate)):
    if (i+1)%8==0:
        print train1_validate[i-7:i+1]

temp = []

raw_input("Press Enter to continue...")

for i in range(0, dataset.shape[0]):
    columns = np.stack(np.split(dataset[i],8,axis=0))
    squares = np.stack(np.split(columns,8,axis=2))
    reshape = squares.reshape(64,900*4)
    temp.append(reshape)

#Flat stanley!
dataset_flat = np.stack(temp)

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.01, verbose=False)
# Gamma is a hyperparameter :D

# We learn the pieces on the first training example
n_samples = dataset.shape[0]

classifier.fit(dataset_flat[0], train1_validate)

#Prediction of first example (image 1 of scotch open seq.)
predicted = classifier.predict(dataset_flat[1])

#Print predicted board in original left to right chessboard format (e.g. train1_validate)

print("The test example image is:")
plt.imshow(dataset[1], cmap=plt.cm.gray_r, interpolation='nearest', aspect='auto')
plt.title('Validation Set')
plt.axis('off')
plt.show()

raw_input("Press Enter to continue...")

print("The predicted values are:")
for i in range(0,len(predicted)):
    if (i+1)%8==0:
        print predicted[i-7:i+1]

raw_input("Press enter to terminate.")
