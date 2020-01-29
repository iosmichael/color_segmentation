import os
import numpy as np
from PIL import Image

def get_data(id):
    img = np.array(Image.open('imgs/{}.png'.format(id)).convert('HSV'))
    img_rgb = np.array(Image.open('imgs/{}.png'.format(id)).convert('RGB'))
    img_lab = np.array(Image.open('imgs/{}.png'.format(id)).convert('YCbCr'))
    img = np.concatenate((img, img_rgb, img_lab),axis=2)
    print(img.shape)
    mask = np.load('masks/{}.npy'.format(id))
    is_stop, not_stop = img[mask], img[~mask]
    np.random.shuffle(is_stop)
    np.random.shuffle(not_stop)
    n, _ = is_stop.shape
    num_to_sample = n
    return is_stop[:num_to_sample,:], not_stop[:4*num_to_sample, :]

def main():
    inds = []
    for f in os.listdir('imgs/'):
        if '.png' in f:
            inds.append(f.split('.png')[0])
    sp_data, nsp_data = [], []
    for id in inds:
        sp, nsp = get_data(id)
        print(sp.shape)
        sp_data.append(sp)
        nsp_data.append(nsp)
    print(len(sp_data))
    print(len(nsp_data))
    sp_data = np.vstack(sp_data)
    nsp_data = np.vstack(nsp_data)
    print(sp_data.shape)
    print(nsp_data.shape)
    np.save("stop_hsvrgbycbcr.npy", sp_data)
    np.save("nstop_hsvrgbycbcr.npy", nsp_data)

if __name__ == '__main__':
    main()
