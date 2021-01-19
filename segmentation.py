#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:55:29 2021

@author: kuangensustech
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img_phase = cv2.imread('original_img/phase.jpeg', cv2.IMREAD_UNCHANGED)
img_fluor= cv2.imread('original_img/fluor.jpeg', cv2.IMREAD_UNCHANGED)
print(img_fluor.shape)
fluor_vals = np.sort(img_fluor.reshape((-1)))
fluor_grads = fluor_vals[1:] - fluor_vals[:-1]
print(np.quantile(fluor_vals, 0.99))
# plt.plot(fluor_vals)
# plt.plot(fluor_grads)
cv2.imshow('fluor', img_fluor)
# cv2.imshow('phase', img_phase)

_, img_fluor = cv2.threshold(img_fluor, thresh=np.mean(fluor_vals) + 2 * np.std(fluor_vals), maxval=255, type=cv2.THRESH_BINARY)

cv2.imshow('fluor_binary', img_fluor)
# cv2.waitKey(0)