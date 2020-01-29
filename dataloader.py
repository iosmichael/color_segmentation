from PIL import Image
import numpy as np
import cv2

class DataLoader(object):

    def __init__(self, is_hsv):
        ext = ''
        if is_hsv:
            ext = '_hsvrgb'
        self.sp_data = np.load('dataset/stop{}.npy'.format(ext))
        self.sp_data = np.hstack((self.sp_data, np.ones((self.sp_data.shape[0], 1))))
        self.nsp_data = np.load('dataset/nstop{}.npy'.format(ext))
        self.nsp_data = np.hstack((self.nsp_data, np.zeros((self.nsp_data.shape[0], 1))))
        self.num = self.sp_data.shape[0]
        print('not stop label: {}'.format(self.nsp_data[:, -1]))
        print('stop label: {}'.format(self.sp_data[:, -1]))
        print('stop shape: {}'.format(self.sp_data.shape))
        print('not stop shape: {}'.format(self.nsp_data.shape))
           
           
    def train_validation_test_split(self):
        np.random.seed(10)
        t, v, ts = self.num * 8 // 10, self.num // 10, self.num //10
        nt, nv, nts = 3 * t, 3 * v, 3 * ts
        t_data = np.vstack((self.sp_data[:t,:], self.nsp_data[:nt, :]))
        v_data = np.vstack((self.sp_data[t:t+v, :], self.nsp_data[nt:nt+nv, :]))
        ts_data = np.vstack((self.sp_data[t+v:, :], self.nsp_data[nt+nv:, :]))
        np.random.shuffle(t_data)
        np.random.shuffle(v_data)
        np.random.shuffle(ts_data)
        self.t_data = t_data
        self.v_data = v_data
        self.ts_data = ts_data
    
    def get_train(self):
        return self.t_data[:, :-1], self.t_data[:, -1]

    def get_validation(self):
        return self.v_data[:, :-1], self.v_data[:, -1]
    
    def get_test(self):
        return self.ts_data[:, :-1], self.ts_data[:, -1]

if __name__ == "__main__":
    dataloader = DataLoader()
    dataloader.train_validation_test_split()
    train_X, train_Y = dataloader.get_train()
    print('input shape: {}'.format(train_X.shape))
    print('label shape: {}'.format(train_Y.shape))
    index = 0

    print('example input: {}'.format(train_X[index, :]))
    print('example label: {}'.format(train_Y[index]))
            
