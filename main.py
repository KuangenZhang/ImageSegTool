import cv2
import numpy as np
import glob
import os

def label_target(img, length = 224, output_dir ='segmented_img', out_img_name = '1'):
    def draw_square(event,x,y, flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            global target_u, target_v
            target_u, target_v = x,y
            upper_left, lower_right = center_to_corners(np.array([target_u, target_v]), length=length)
            cv2.rectangle(img, tuple(upper_left.tolist()),
                          tuple(lower_right.tolist()), color=(0, 255, 0), thickness=5)
            # print('u: {}, v: {}'.format(target_u, target_v))

    global target_u, target_v
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_square)
    img_original= np.copy(img)
    while(1):
        while (1):
            cv2.imshow('image', img)
            k = cv2.waitKey(20) & 0xFF
            if k == ord('s') or k == 27:  # 's': save and continue; Esc: save and stop
                out_img_full_name = '{}/u_{:d}_v_{:d}_{}'.format(output_dir, target_u, target_v, out_img_name)
                upper_left, lower_right = center_to_corners(np.array([target_u, target_v]), length=length)
                segmented_img = img_original[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
                if length == segmented_img.shape[0] and length == segmented_img.shape[1]:
                    cv2.imwrite(out_img_full_name, segmented_img)
                img = np.copy(img_original) # refresh the img
                break
        if k == 27:  # Esc key to stop labeling the current image
            break

def center_to_corners(center, length = 224):
    return center - int(length / 2), center + int(length / 2)

def label_file_in_folder(img_dir = 'original_img'):
    img_name_list = glob.glob('original_img/**/*.tif', recursive=True)
    for img_name in img_name_list:
        img = cv2.imread(img_name)
        output_dir = '{}/segmented_img'.format(os.path.dirname(img_name))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        label_target(img, output_dir = output_dir, out_img_name = os.path.basename(img_name))

if __name__ == '__main__':
    label_file_in_folder()