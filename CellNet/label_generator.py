import os
import numpy as np
import pandas as pd
import cv2

dir = os.path.join(os.getcwd(), 'test')

def Flip_Image(dir, flip_dir):
    img = cv2.imread(dir, 1)
    flip = cv2.flip(img, flip_dir)
    cv2.imwrite(os.path.join(os.path.dirname(dir), str(flip_dir) + 'f' + os.path.basename(dir)), flip)

if __name__ == '__main__':
    cells = [os.path.join(dir, 'cell', cell) for cell in os.listdir(os.path.join(dir, 'cell'))]
    ncells = [os.path.join(dir, 'ncell', ncell) for ncell in os.listdir(os.path.join(dir, 'ncell'))]
    # for cell in cells:
    #     Flip_Image(cell, 1)
    #     # Flip_Image(cell, 0)
    # for ncell in ncells:
    #     Flip_Image(ncell, 1)
    #     # Flip_Image(ncell, 0)
    cells = [os.path.join(dir, 'cell', cell) for cell in os.listdir(os.path.join(dir, 'cell'))]
    ncells = [os.path.join(dir, 'ncell', ncell) for ncell in os.listdir(os.path.join(dir, 'ncell'))]
    labels = np.concatenate((np.ones(len(cells)), np.zeros(len(ncells))))
    labels = np.array(labels, dtype=np.uint8)
    summary = pd.DataFrame({'dir': np.concatenate((cells, ncells)), 
                            'label': labels})
    summary.to_csv(os.path.join(dir, "labels.csv"), header = 0, index = 0)