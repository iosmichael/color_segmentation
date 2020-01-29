import numpy as np
from classifier import LogisticRegression
from dataloader import DataLoader
import matplotlib
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB

'''
classifier that discriminates the stop color from non-stop color
'''

config = {
          'epoch': 0,
          'lr': 0.0001,
          'model': 'logistic',
          'minibatch': 256
        }
'''
save best model
'''
def train():
    model = LogisticRegression((4,1))
    dataloader = DataLoader()
    dataloader.train_validation_test_split()
    train_X, train_Y = dataloader.get_train()
    val_X, val_Y = dataloader.get_validation()
    test_X, test_Y = dataloader.get_test()
    return train_X, train_Y

from detection import detection, shape_similarity 
import argparse

if __name__ == "__main__":

    # model = train()
    train_X, train_Y = train()
    gnb = GaussianNB()
    test_img = np.array(Image.open('trainset/10.jpg'))
    test_X = test_img.reshape(-1, 3)
    y = gnb.fit(train_X, train_Y).predict(test_X)

    #transform(np.array(test_img), model = LogisticRegression((4,1)))
    plt.imshow(y.reshape(test_img.shape[:2]), cmap='gray')
    plt.show()
    mask = y.reshape(test_img.shape[:2])
    bitmap = detection(mask)
#    from skimage.measure import find_countour
    shape_similarity(mask)
