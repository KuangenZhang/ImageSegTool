#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:55:29 2021

@author: kuangensustech
"""
from skimage import measure
import cv2
import os
import numpy as np
import glob
import matplotlib.pyplot as plt


def draw_cell_contour():
    img_phase = cv2.imread('original_img/phase.jpeg', cv2.IMREAD_UNCHANGED)
    img_phase_rgb = cv2.cvtColor(img_phase, cv2.COLOR_GRAY2RGB)
    filtered_fluor_label_prop_list, _, _, _ = recognize_fluor_labels()
    radius = 30
    contour_list = []
    for i in range(len(filtered_fluor_label_prop_list)):
        filtered_fluor_label_prop = filtered_fluor_label_prop_list[i]
        (r_c, c_c) = filtered_fluor_label_prop.centroid
        (r_c, c_c) = (int(r_c), int(c_c))
        _, _, _, _, contour = \
            recognize_cell_border(img_phase[r_c - radius:r_c + radius, c_c - radius:c_c + radius])
        cv2.drawContours(img_phase_rgb, contour + np.asarray([[c_c - radius, r_c - radius]]), -1, (0, 255, 0), 5)
    cv2.imshow('img_phase_rgb', img_phase_rgb)
    cv2.imwrite('results/img_phase_rgb.jpg', img_phase_rgb)
    cv2.waitKey(0)


def recognize_fluor_labels():
    '''Segment phase image based on the center point shown in the flour image.'''
    img_fluor_origin = cv2.imread('original_img/fluor.jpeg', cv2.IMREAD_UNCHANGED)
    _, img_fluor = cv2.threshold(img_fluor_origin, thresh=np.mean(img_fluor_origin) + 2 * np.std(img_fluor_origin),
                                 maxval=255, type=cv2.THRESH_BINARY)

    img_fluor = cv2.medianBlur(img_fluor, ksize=7)
    fluor_labels = measure.label(img_fluor, background=0, neighbors=8, connectivity=2)
    fluor_labels_prop_list = measure.regionprops(fluor_labels)
    area_list = []

    for fluor_labels_prop in fluor_labels_prop_list:
        area_list.append(fluor_labels_prop.filled_area)
    area_vec = np.array(area_list)
    fluor_labels_prop_list = np.array(fluor_labels_prop_list)
    filtered_fluor_label_prop_list = fluor_labels_prop_list
    # filtered_fluor_label_prop_list = fluor_labels_prop_list[area_vec < np.mean(area_vec) + np.std(area_vec)]
    return filtered_fluor_label_prop_list, img_fluor_origin, img_fluor, fluor_labels

def recognize_cell_border(img_phase):
    _, img_phase_binary = cv2.threshold(img_phase, thresh=np.mean(img_phase),
                                 maxval=255, type=cv2.THRESH_BINARY)
    (rows, cols) = img_phase.shape
    img_mask = np.zeros(img_phase.shape, dtype= np.uint8)
    cv2.circle(img_mask, center=(int(cols / 2), int(rows / 2)), radius=int(cols / 5), thickness=int(cols / 2.5),
               color=1)
    img_phase_binary = (img_phase_binary * img_mask).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    # Repeat the erosion and dilation by changing iterations.
    img_dilate = cv2.dilate(img_phase_binary, kernel, iterations=5)
    img_dilate = cv2.medianBlur(img_dilate, ksize=7)

    img_dilate_no_border = np.zeros(img_phase.shape, dtype= np.uint8)
    img_dilate_no_border[1:-1, 1:-1] = img_dilate[1:-1, 1:-1] # avoid connecting with border

    img_erode = cv2.erode(img_dilate_no_border, kernel, iterations=5)

    contours, _ = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img_phase = cv2.cvtColor(img_phase, cv2.COLOR_GRAY2RGB)
    for contour in contours:
        cv2.drawContours(img_phase, contour, -1, (0, 255, 0), 5)
    return img_phase, img_phase_binary, img_dilate_no_border, img_erode, contours[0]


def segment_phase_img():
    '''Segment phase image based on the center point shown in the flour image.'''
    img_phase = cv2.imread('original_img/phase.jpeg', cv2.IMREAD_UNCHANGED)
    filtered_fluor_label_prop_list, img_fluor_origin, img_fluor, fluor_labels = recognize_fluor_labels()
    img_phase_rgb = cv2.cvtColor(img_phase, cv2.COLOR_GRAY2RGB)
    img_fluor_rgb = cv2.cvtColor(img_fluor, cv2.COLOR_GRAY2RGB)
    segmented_phase_img_dir = 'results/segmented_phase_img'
    if not os.path.exists(segmented_phase_img_dir):
        os.mkdir(segmented_phase_img_dir)
    radius = 30
    for i in range(len(filtered_fluor_label_prop_list)):
        filtered_fluor_label_prop = filtered_fluor_label_prop_list[i]
        (r_c, c_c) = filtered_fluor_label_prop.centroid
        (r_c, c_c) = (int(r_c), int(c_c))
        cv2.circle(img_phase_rgb, (c_c, r_c), radius=radius, color=[0, 255, 0])
        cv2.circle(img_fluor_rgb, (c_c, r_c), radius=radius, color=[0, 255, 0])
        cv2.imwrite('{}/{}.png'.format(segmented_phase_img_dir, i), img_phase[r_c-radius:r_c+radius, c_c-radius:c_c+radius])
    img_list = [img_fluor_origin, fluor_labels, img_fluor_rgb, img_phase_rgb]
    img_names = ['img_fluor_origin', 'fluor_labels', 'img_fluor_rgb', 'img_phase_rgb']
    plot_images(img_list, img_names, is_save_img = False)
    plt.savefig('results/segmented_phase_img.jpg')
    plt.show()


def recognize_all_cell_borders():
    '''Recognize the border of cell from segmented phase images'''
    img_names = glob.glob('results/segmented_phase_img/*.png')
    for img_name in img_names:
        img_phase = cv2.imread(img_name, flags=cv2.IMREAD_UNCHANGED)
        img_phase, img_phase_binary, img_dilate_no_border, img_erode, contours = recognize_cell_border(img_phase)
        img_list = [img_phase, img_phase_binary, img_dilate_no_border, img_erode]
        img_names = ['img_phase', 'img_phase_binary', 'img_dilate_no_border', 'img_erode']
        plot_images(img_list, img_names, is_save_img=False)
        plt.show()



def plot_images(img_list, img_names, is_save_img = False):
    plt.figure(figsize=(16, 16))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(img_list[i])
        plt.title(img_names[i])
        plt.axis('off')
        if is_save_img:
            cv2.imwrite('results/{}.jpg'.format(img_names[i]), img_list[i])
    plt.tight_layout()



if __name__ == '__main__':
    # segment_phase_img()
    # recognize_cell_border()
    draw_cell_contour()