from pywinauto import application
import pyautogui
import time
import os
import sys
import re
from PIL import Image, ImageQt, ImageFilter
import pytesseract
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.transforms import ToTensor, Compose
from torchvision.models import resnet18
from scipy import ndimage as ndi
import win32gui
from PyQt5.QtWidgets import QApplication
import psutil
import shutil
import cv2

START = 101
SLEEP_TIME = 1
DIR = os.getcwd()
if(not os.path.exists(os.path.join(DIR, 'log'))):
    os.mkdir(os.path.join(DIR, 'log'))
TOTAL_NUM = len([i for i in os.listdir(os.path.dirname(DIR)) if i.endswith('.mat')])

device = 'cpu'
transform_size = 224
transform = Compose([ToTensor()])

def Get_Model(model_path):
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2, bias = False)
    model.load_state_dict(torch.load(model_path))
    return model

class Pywin(object):
    SLEEP_TIME = 1
    def __init__(self):
        self.app = application.Application(backend='uia')
        
    def run(self, tool_name):
        self.app.start(tool_name)
        time.sleep(self.SLEEP_TIME)
    
    def connect(self, window_name):
        self.app.connect(title = window_name)
        time.sleep(self.SLEEP_TIME)
    
    def max_window(self, window_name):
        self.app[window_name].maximize()
        time.sleep(self.SLEEP_TIME)
    
    def min_window(self, window_name):
        self.app[window_name].minimize()
        time.sleep(self.SLEEP_TIME)

    def send_k(self, window_name, key_name):
        self.app[window_name].type_keys(key_name)
        time.sleep(self.SLEEP_TIME)

    def close(self, window_name):
        self.app[window_name].close()
        time.sleep(self.SLEEP_TIME)

def Get_Screenshot(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    if(hwnd != None):
        app = QApplication(sys.argv)
        screen = QApplication.primaryScreen()
        qimg = screen.grabWindow(hwnd).toImage()
        img = ImageQt.fromqimage(qimg)
        return img

def Crop_Title(temp_dir, oimg):
    if(temp_dir):
        img = cv2.imread(temp_dir, 0)
    else:
        img = np.array(oimg.convert('L'))
        img[np.where(img >= 230)] = 255
    gray_img = 255 - img
    by_row = np.sum(gray_img, axis = 1)
    xind = np.where(by_row != 0)
    xind_0 = np.where(by_row == 0)[0]
    xl = xind[0][0]
    tmp = np.where(xind_0 > xl)[0]
    xr = xind_0[tmp[0]]
    if(xl >= 10):
        xl = xl - 10
    else:
        xl = 0

    by_col = np.sum(gray_img, axis = 0)
    yind = np.where(by_col != 0)
    yu = yind[0][0]
    yd = yind[0][-1]
    crop = img[xl:xr, yu:yd]
    print(crop.shape)
    out = cv2.resize(crop, None, fx = 2, fy = 2, interpolation = cv2.INTER_LANCZOS4)
    return(out)

def Current_Cell(img):
    # img = img.crop((20, 80) + img.size).convert('L')
    # img = img.convert('L')
    # imout = img.point(lambda i : i < 200 and 255)
    # imout = img.filter(ImageFilter.SHARPEN).filter(ImageFilter.SMOOTH).filter(ImageFilter.CONTOUR)
    en = pytesseract.image_to_string(img)
    temp = re.match(r'Object: (?P<id>\d+) out of (?P<tot>\d+)', en)
    if temp != None:
        cur = temp.group('id')
        tot = temp.group('tot')
        print(temp.group())
        if int(cur) < int(tot):
            return True
        else:
            return False
    else:
        return 'skip'

def Crop_Image(temp_dir, color):
    if(color):
        img = cv2.imread(temp_dir, 1)
        gray_img = 255 - img[:, :, 0]
    else:
        img = cv2.imread(temp_dir, 0)
        gray_img = 255 - img
    
    by_row = np.sum(gray_img, axis = 1)
    xind = np.where(by_row != 0)[0]
    xind_0 = np.where(by_row == 0)[0]
    tmp1 = xind[0]
    tmp2 = np.where(xind_0 > tmp1)[0]
    tmp3 = xind_0[tmp2[0]]
    tmp4 = np.where(xind > tmp3)[0]
    xl = xind[tmp4[0]]
    xr = xind[-1]

    by_col = np.sum(gray_img, axis = 0)
    yind = np.where(by_col != 0)
    yu = yind[0][0]
    yd = yind[0][-1]
    if(color):
        crop = img[xl:xr, yu:yd, :]
    else:
        crop = img[xl:xr, yu:yd]
    imgout = cv2.resize(crop, (transform_size, transform_size))
    return Image.fromarray(imgout)

def test(model, img):
    with torch.no_grad():
        imgt = transform(img)
        imgt.unsqueeze_(0)
        imgt = imgt.to(device)
        pred = model(imgt)
    return pred

def Crop_Area(temp_dir):
    img = cv2.imread(temp_dir, 1)
    gray_img = 255 - img[:, :, 0]
    by_row = np.sum(gray_img, axis = 1)

    xind = np.where(by_row != 0)[0]
    xind_0 = np.where(by_row == 0)[0]
    tmp1 = xind[0]
    tmp2 = np.where(xind_0 > tmp1)[0]
    tmp3 = xind_0[tmp2[0]]
    tmp4 = np.where(xind > tmp3)[0]
    xl = xind[tmp4[0]]
    xr = xind[-1]

    by_col = np.sum(gray_img, axis = 0)
    # plt.plot(by_col)
    yind = np.where(by_col != 0)[0]
    yind_0 = np.where(by_col == 0)[0]
    tmp1 = yind[0]
    tmp2 = np.where(yind_0 > tmp1)[0]
    tmp3 = yind_0[tmp2[0]]
    tmp4 = np.where(yind > tmp3)[0]
    yu = yind[tmp4[0]]
    yd = yind[-1]
    crop = img[xl:xr, yu:yd, :]
    return crop

def Area_Calc(temp_dir):
    mu = 950
    sigma = 389
    factor = 3
    low_thr = 300
    img = Crop_Area(temp_dir)
    ratio = 2048 / img.shape[1]
    blue = img[:, :, 0]
    red = img[:, :, 2]
    contour = np.array((blue > 100) * (red < 100), dtype = np.uint8)
    filled_binary = ndi.binary_fill_holes(contour)
    area = filled_binary.sum() * (ratio ** 2)

    if((abs(mu - area) <= factor * sigma) and (area >= low_thr)):
        return True
    else:
        return False
    
def Crop_Screenshot(oimg):
    img = np.array(oimg.convert('L'))
    img[np.where(img >= 230)] = 255
    gray_img = 255 - img
    by_row = np.sum(gray_img, axis = 1)
    xind = np.where(by_row != 0)[0]
    xind_0 = np.where(by_row == 0)[0]
    tmp1 = xind[0]
    tmp2 = np.where(xind_0 > tmp1)[0]
    tmp3 = xind_0[tmp2[0]]
    tmp4 = np.where(xind > tmp3)[0]
    xl = xind[tmp4[0]]
    xr = xind[-1]

    by_col = np.sum(gray_img, axis = 0)
    yind = np.where(by_col != 0)
    yu = yind[0][0]
    yd = yind[0][-1]

    crop = oimg.crop((yu, xl, yd, xr))
    imgout = crop.resize((transform_size, transform_size))
    return imgout

def Decision_Maker(temp_dir, model):
    if(temp_dir):
        title = Crop_Title(temp_dir)
        flag = Current_Cell(title)
        img = Crop_Image(temp_dir, True)
    else:
        img = Get_Screenshot('Figure 2')
        img = img.crop((0, 100, img.size[0], img.size[1]))
        title = Crop_Title(temp_dir = None, oimg = img)
        flag = Current_Cell(title)
        if(flag == 'skip'):
            return [False, True]
        imgout = Crop_Screenshot(img)
    pred = test(model, imgout)
    if(pred.argmax(1) == 1):
        return [True, flag]
    else:
        return [False, flag]

def Left_Window(window_name, app):
    app.max_window(window_name)
    pyautogui.click(100, 1)
    time.sleep(SLEEP_TIME)
    pyautogui.hotkey('win', 'left')
    time.sleep(SLEEP_TIME)

def Save_image(window_name, app, save_dir):
    pyautogui.click(18, 45)
    time.sleep(SLEEP_TIME)
    pyautogui.click(57, 154)
    time.sleep(SLEEP_TIME)
    dlg = app.app[window_name]
    figure_dir = save_dir
    dlg['Save As'].child_window(auto_id = '1001', control_type = 'Edit').set_text(figure_dir)
    time.sleep(SLEEP_TIME)
    dlg['Save As'].child_window(auto_id = '1', control_type = 'Button').click()
    time.sleep(SLEEP_TIME)


if __name__ == '__main__':
    t0 = time.asctime()
    # pyautogui.hotkey('win', 'right')
    # time.sleep(SLEEP_TIME)
    app = Pywin()
    app.connect('MATLAB R2019b - academic use')
    model_path = os.path.join(DIR, 'model.pth')
    model = Get_Model(model_path)
    model.to(device)
    model.eval()

    for i in range(TOTAL_NUM):
        if i < START:
            # pyautogui.press('right')
            # time.sleep(SLEEP_TIME * 5)
            continue
        Left_Window('Figure 1', app)
        x, y = pyautogui.locateCenterOnScreen('but.png')
        pyautogui.click(x, y)
        time.sleep(SLEEP_TIME)
        # app.min_window('Figure 2')
        Left_Window('Figure 2', app)

        while True:
            # figure1_dir = os.path.join(DIR, 'figure1.bmp')
            # app.max_window('Figure 1')
            # Save_image('Figure 1', app, figure1_dir)

            # Left_Window('Figure 1', app)
            # app.max_window('Figure 2')
            # Left_Window('Figure 2', app)
            # figure2_dir = os.path.join(DIR, 'figure2.bmp')
            # Save_image('Figure 2', app, figure2_dir)
            decision = Decision_Maker(temp_dir = None, model = model)
            # decision[0] = decision[0] and Area_Calc(figure1_dir)
            print("Cell: %r || Continue: %r" % (decision[0], decision[1]))
            pyautogui.click(100, 100)
            if(decision[0]):
                pyautogui.press('up')
            else:
                pyautogui.press('down')
            time.sleep(SLEEP_TIME)
            # os.remove(figure2_dir)
            # os.remove(figure1_dir)

            if(not decision[1]):
                app.close('Figure 2')
                break
        
        # app.max_window('Figure 1')
        log_dir = os.path.join(DIR, 'log', str(i).zfill(4) + '.bmp')
        Save_image('Figure 1', app, log_dir)
        pyautogui.click(100, 100)
        time.sleep(SLEEP_TIME)
        pyautogui.press('delete')
        pyautogui.click(100, 100)
        time.sleep(SLEEP_TIME)
        pyautogui.press('delete')
        time.sleep(SLEEP_TIME * 5)
        x2, y2 = pyautogui.locateCenterOnScreen('save.png')
        pyautogui.click(x2, y2)
        time.sleep(SLEEP_TIME)
        pyautogui.click(100, 100)
        time.sleep(SLEEP_TIME)
        pyautogui.press('right')
        time.sleep(SLEEP_TIME * 10)

    t1 = time.asctime()
    print("Start:", t0)
    print("End:", t1)
    
