from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import os, re

matplotlib.use('TkAgg')

from roipoly import RoiPoly

def label_image(img_path, img_id):
    # Create image
    img = Image.open('trainset/{}.jpg'.format(img_id))

    # Show the image
    fig = plt.figure()
    plt.imshow(img)

    # Let user draw first ROI
    roi = RoiPoly(color='r', fig=fig)
    mask = roi.get_mask(np.array(img)[:, :, 0])
    plt.imshow(mask)
    plt.show()
    print(mask)
    np.save('dataset/masks/{}.npy'.format(img_id), mask)
    img.save('dataset/imgs/{}.png'.format(img_id))

for f in os.listdir('trainset'):
    if '.jpg' in f:
        id = f.split('.jpg')[0]
        if int(id) <= 100:
            label_image('', id)
        

