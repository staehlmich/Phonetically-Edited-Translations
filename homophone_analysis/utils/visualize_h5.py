#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
import imageio

def main():

    f = h5py.File('/home/user/staehli/master_thesis/data/MuST-C/test.audio.h5', 'r')

    dset = f["audio_682"]
    data = np.array(dset[:, :])
    file = 'test2.png'  # or .jpg
    imageio.imwrite(file, data)

if __name__ == "__main__":
    main()