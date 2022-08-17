from pywinauto import application
import pyautogui
import time
import os
import sys
import re
from PIL import Image, ImageQt, ImageFilter
import pytesseract
import numpy as np
from scipy import ndimage as ndi
import win32gui
from PyQt5.QtWidgets import QApplication
import psutil
import shutil
import cv2

SLEEP_TIME = 1
START = 1415
PARENT_DIR = os.getcwd()
OUT_DIR = os.path.join(PARENT_DIR, 'train')
TOTAL_NUM = len([i for i in os.listdir(os.path.dirname(PARENT_DIR)) if i.endswith('.mat')])
VOTE = ['{DOWN}', '{UP}']

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
    
    def input(self, window_name, controller, content):
        self.app[window_name][controller].TypeKeys(content)
        time.sleep(self.SLEEP_TIME)

    def print_prop(self, window_name):
        self.app[window_name].print_control_identifiers()

def Get_Screenshot(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    if(hwnd != None):
        app = QApplication(sys.argv)
        screen = QApplication.primaryScreen()
        qimg = screen.grabWindow(hwnd).toImage()
        img = ImageQt.fromqimage(qimg)
        return img


def Crop_Image(temp_dir):
    img = cv2.imread(temp_dir, 0)
    gray_img = 255 - img
    by_row = np.sum(gray_img, axis = 1)
    xind = np.where(by_row != 0)
    xind_0 = np.where(by_row == 0)[0]
    xl = xind[0][0]
    tmp = np.where(xind_0 > xl)[0]
    xr = xind_0[tmp[0]]
    xl = xl - 10

    by_col = np.sum(gray_img, axis = 0)
    yind = np.where(by_col != 0)
    yu = yind[0][0]
    yd = yind[0][-1]
    crop = img[xl:xr, yu:yd]
    print(crop.shape)
    out = cv2.resize(crop, None, fx = 2, fy = 2, interpolation = cv2.INTER_LANCZOS4)
    return(out)

def Crop_Title(oimg):
    img = np.array(oimg.convert('L'))
    img[np.where(img >= 230)] = 255
    gray_img = 255 -img
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

def Current_Cell(oimg):
    # img = img.crop((20, 80) + img.size).convert('L')
    # img = img.convert('L')
    # imout = img.point(lambda i : i < 200 and 255)
    # imout = img.filter(ImageFilter.SHARPEN).filter(ImageFilter.SMOOTH).filter(ImageFilter.CONTOUR)
    img = Crop_Title(oimg)
    en = pytesseract.image_to_string(img)
    temp = re.match(r'Object: (?P<id>\d+) out of (?P<tot>\d+)', en)
    cur = temp.group('id')
    tot = temp.group('tot')
    print(temp.group())
    if int(cur) < int(tot):
        return True
    else:
        return False

def Decide_By_Me(temp_dir, screenshot):
    # img = Get_Screenshot('Figure 2')
    if(not screenshot):
        img = Crop_Image(temp_dir)
    else:
        img = Get_Screenshot('Figure 2')
        img = img.crop((0, 100, img.size[0], img.size[1]))
    flag = Current_Cell(img)
    # process_list = []
    # for proc in psutil.process_iter():
    #     process_list.append(proc)
    # img.show('Prev')
    pyautogui.click(1380, 980)
    time.sleep(SLEEP_TIME)
    print('Is this a cell?[y/n]', end = ':')
    select = input()
    img.save(temp_dir)
    # for proc in psutil.process_iter():
    #     if not proc in process_list:
    #         proc.kill()
    if(select == 'y'):
        return [True, flag]
    else:
        return [False, flag]
    

if __name__ == '__main__':
    app = Pywin()
    app.connect('MATLAB R2019b - academic use')
    d = START
    for i in range(TOTAL_NUM):
        app.max_window('Figure 1')
        pyautogui.click(100, 1)
        time.sleep(SLEEP_TIME)
        pyautogui.hotkey('win', 'left')
        time.sleep(SLEEP_TIME)
        # app.send_k('Figure 1', ("{RWINff down}", "{LEFT down}", "{LEFT up}","{RWIN up}"))
        x, y = pyautogui.locateCenterOnScreen('but.png')
        pyautogui.click(x, y)
        time.sleep(SLEEP_TIME)
        app.max_window('Figure 2')
        pyautogui.click(100, 1)
        time.sleep(SLEEP_TIME)
        pyautogui.hotkey('win', 'left')
        time.sleep(SLEEP_TIME)
        # app.send_k('Figure 2', ("{RWIN down}", "{LEFT down}", "{LEFT up}","{RWIN up}"))
        while True:
            # time.sleep(SLEEP_TIME)
            # pyautogui.click(18, 45)
            # time.sleep(SLEEP_TIME)
            # pyautogui.click(57, 154)
            # dlg = app.app['Figure 2']
            temp_dir = os.path.join(OUT_DIR, str(d).zfill(5) + '.bmp')
            # dlg['Save As'].child_window(auto_id = '1001', control_type = 'Edit').set_text(temp_dir)
            # time.sleep(SLEEP_TIME)
            # dlg['Save As'].child_window(auto_id = '1', control_type = 'Button').click()
            # time.sleep(SLEEP_TIME)
            
            # img = Image.open(temp_dir)
            decision = Decide_By_Me(temp_dir, True)
            print("Cell: %r || Continue: %r" % (decision[0], decision[1]))
            if(decision[0]):
                shutil.move(temp_dir, os.path.join(OUT_DIR, 'cell', str(d).zfill(5) + '.bmp'))
                # new_image = os.path.join(OUT_DIR, 'cell', str(d).zfill(5)+'.tif')
            else:
                shutil.move(temp_dir, os.path.join(OUT_DIR, 'ncell', str(d).zfill(5) + '.bmp'))
                # new_image = os.path.join(OUT_DIR, 'ncell', str(d).zfill(5)+'.tif')
            # app.max_window('Figure 2')
            d = d + 1
            time.sleep(SLEEP_TIME)
            pyautogui.click(100, 100)
            time.sleep(SLEEP_TIME)
            if(decision[0]):
                pyautogui.press('up')
                # app.send_k('Figure 2', "{RIGHT}")
            else:
                pyautogui.press('down')
            time.sleep(SLEEP_TIME)
            if(not decision[1]):
                app.close('Figure 2')
                break
        print("%d out of %d finished." % (i, TOTAL_NUM))
        app.max_window('Figure 1')
        pyautogui.click(100, 100)
        time.sleep(SLEEP_TIME)
        pyautogui.press('right')
        

