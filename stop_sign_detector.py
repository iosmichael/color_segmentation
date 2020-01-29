'''
ECE276A WI20 HW1
Stop Sign Detector
'''

import os, cv2
import numpy as np
from skimage.measure import label, regionprops
from dataloader import DataLoader
# from sklearn.naive_bayes import GaussianNB
from gaussian_naive_bayes import GaussianNaiveBayes
from detection import detection
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


threshold = 100

class StopSignDetector():
    def __init__(self):
        '''
            Initilize your stop sign detector with the attributes you need,
            e.g., parameters of your classifier
        '''
        
        self.gnb = GaussianNaiveBayes()
        if not self.gnb.is_fitted:
            dataloader = DataLoader(is_hsv=True)
            dataloader.train_validation_test_split()
            trainX, trainY = dataloader.get_train()
            self.gnb.fit(trainX, trainY)
            valX, valY = dataloader.get_validation()
            testX, testY = dataloader.get_test()
            print('train accuracy: {}'.format(self.gnb.get_accuracy(trainX, trainY)))
            print('validation accuracy: {}'.format(self.gnb.get_accuracy(valX, valY)))
            print('test accuracy: {}'.format(self.gnb.get_accuracy(testY, testY)))

    def segment_image(self, img, vis=True):
        '''
            Obtain a segmented image using a color classifier,
            e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
            call other functions in this class if needed
            
            Inputs:
                img - original image
            Outputs:
                mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
        '''
        seg_img = np.concatenate((np.array(img.convert('HSV')), np.array(img.convert('RGB'))), axis=2)
        
        predX = seg_img.reshape(-1,3+3)
        y = self.gnb.predict(predX)
        if vis:
            plt.imshow(y.reshape(seg_img.shape[:2]), cmap='gray')
            plt.show()
        mask_img = y.reshape(seg_img.shape[:2])
        return mask_img

    def get_bounding_box(self, img, vis=True):
        '''
            Find the bounding box of the stop sign
            call other functions in this class if needed
            
            Inputs:
                img - original image
            Outputs:
                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
                where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
                is from left to right in the image.
                
            Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
        '''
        mask_img = self.segment_image(img)
        # dfs to union patches
        bitmaps = detection(mask_img)
        boxes = []
        for patch in bitmaps:
            bitmap = patch[0]
            for region in regionprops(bitmap.astype(np.int)):
                # skip small images
                if region['Area'] < threshold:
                    continue
                # fit poly to make sure the shape is desirable

                # draw rectangle around segmented coins
                if vis:
                    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
                    ax.imshow(bitmap.astype(np.float), cmap='gray')
                    minr, minc, maxr, maxc = region['BoundingBox']
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
                    plt.show()
                boxes.append((minc, minr, maxc, maxr))
        return boxes

from PIL import Image

if __name__ == '__main__':
    folder = "trainset"
    my_detector = StopSignDetector()
    for filename in os.listdir(folder):
    # for i in range(100):
    #    fname = '{}.jpg'.format(i)
    #    if not os.path.isfile(os.path.join(folder, fname)):
    #        continue
        # read one test image
        # img = cv2.imread(os.path.join(folder,filename))
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img = Image.open(os.path.join(folder, filename))

        h, w = np.array(img).shape[:2]
        min_val = 300
        scale = min_val / min(w, h)
        img = img.resize((int(w * scale), int(h * scale)), resample=Image.BILINEAR)
        # Display results:
        # (1) Segmented images
        print(filename)
        mask_img = my_detector.segment_image(img)
        # (2) Stop sign bounding box
        boxes = my_detector.get_bounding_box(img)
        print(boxes)
        # The autograder checks your answers to the functions segment_image() and get_bounding_box()
        # Make sure your code runs as expected on the testset before submitting to Gradescope
