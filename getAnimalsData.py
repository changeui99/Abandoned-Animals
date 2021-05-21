import win32gui, pydirectinput, cv2, time
import pytesseract
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageGrab

winlist = []

def enum_win(hwnd, result):
    win_text = win32gui.GetWindowText(hwnd)
    winlist.append((hwnd, win_text))

def findwindow():
    toplist = []
    result = None
    win32gui.EnumWindows(enum_win, toplist)

    for (hwnd, win_text) in winlist:
        if 'NoxPlayer' in win_text:
            result = hwnd
            break

    return result

def isEnd(x, y) :
    image = np.array(ImageGrab.grab(bbox=(x + 50, y + 458, x + 529, y + 480)))
    return ((image == 255).all())

def splitToPart(s) :
    if ('：' in s) :
        return s.split('：')
    else :
        return s.split(':')

def getData(x, y) :
    result = {
        'state': np.NAN,
        'kind': np.NAN,
        'sex': np.NAN,
        'neutralization': np.NAN,
        'color': np.NAN,
        'birth':  np.NAN,
        'weight': np.NAN,
        'number': np.NAN,
        'period': np.NAN,
        'location': np.NAN,
        'feature': np.NAN,
        'center': np.NAN,
        'department': np.NAN
    }
    image = np.array(ImageGrab.grab(bbox=(x + 10, y+475, x + 599, y + 510)))
    mask = (image >= 190).all(axis=2)
    mask2 = (image < 190).any(axis=2)
    image[mask] = [0, 0, 0]
    image[mask2] = [255, 255, 255]

    retry = 0

    while retry < 10 :
        text = pytesseract.image_to_string(image[1:, 1:72, :], lang='kor', config='-c preserve_interword_space=1 --psm 4').replace("\n", "")
        #종료(자연사) 종료(안락사) 완료(귀가) 완료(입양) 보호중 공고중

        if ('공고' in text):
            result['state'] = "공고중"
            break
        elif ('보호중' in text):
            result['state'] = "보호중"
            break
        else :
            text = pytesseract.image_to_string(image[1:, 1:106, :], lang='kor',config='-c preserve_interword_space=1 --psm 4').replace("\n", "")

            if ('종' in text and '료' in text) :
                print(text)
                if ('자' in text):
                    result['state'] = "종료(자연사)"
                    break
                elif ('안' in text):
                    result['state'] = "종료(안락사)"
                    break
                elif ('기' in text):
                    result['state'] = "종료(기증)"
                    break
                elif ('방' in text) :
                    result['state'] = "종료(방사)"
                    break
            elif ('완' in text or '왼' in text and '료' in text) :
                if ('귀가' in text) :
                    result['state'] = "완료(귀가)"
                    break
                elif ('입양' in text) :
                    result['state'] = "완료(입양)"
                    break
            else :
                if (retry == 9) :
                    print(text)
                    return None
                    plt.imshow(image[1:, 1:106, :])
                    plt.show()

                    a = input("text")

                    result['state'] = a
                retry += 1
                print("retry : ", retry)
                time.sleep(1)

    image = np.array(ImageGrab.grab(bbox=(x, y + 528, x + 599, y + 810)))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    text = pytesseract.image_to_string(image, lang='kor', config='-c preserve_interword_space=1 --psm 4').split("\n")
    for t in text :
        if ('[개]' in t) :
            result['kind'] = t
        elif ('암컷' in t or '수컷' in t) :
            for idx, tempt in enumerate(t.split('/')) :
                if (idx == 0) :
                    result['sex'] = tempt[:2]
                    if ('×' in tempt) :
                        result['neutralization'] = 0
                    if ('0' in tempt) :
                        result['neutralization'] = 1
                if (idx == 1) :
                    result['color'] = tempt.strip()
                if ('년생' in tempt) :
                    result['birth'] = tempt.replace(" ", "")[:4]

                if (idx == 3):
                    if ('(' in tempt) :
                        result['weight'] = tempt.split('(')[0].replace(" ", "")
                    elif ('0<' in tempt) :
                        result['weight'] = tempt.split('0<')[0].replace(" ", "")
                    elif ('<' in tempt) :
                        result['weight'] = tempt.split('<')[0].replace(" ", "")
                    elif ('09' in tempt) :
                        result['weight'] = tempt.split('09')[0].replace(" ", "")
        elif ('공고번호' in t) :
            try:
                result['number'] = splitToPart(t)[1].strip()
            except IndexError :
                pass
        elif ('공고기간' in t) :
            result['period'] = splitToPart(t)[1].strip()
        elif ('발견장소' in t) :
            try :
                result['location'] = splitToPart(t)[1].strip()
            except IndexError :
                pass
        elif ('특이사항' in t) :
            result['feature'] = splitToPart(t)[1].strip()
        elif ('보호센터' in t) :
            result['center'] = splitToPart(t)[1].split("(")[0].strip()
        elif ('담당부서' in t) :
            result['department'] = splitToPart(t)[1].split("(")[0].strip()

    print(result)

    return pd.Series(
        [result['state'], result['kind'], result['sex'], result['neutralization'], result['color'], result['birth'], result['weight'],
         result['number'], result['period'], result['location'], result['feature'], result['center'], result['department']],
        ['state', 'kind', 'sex', 'neutralization', 'color', 'birth', 'weight', 'number', 'period', 'location', 'feature', 'center', 'department']
    )


if (__name__ == '__main__') :
    data_list = []

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    hwnd = findwindow()
    if (hwnd) :
        count = 0
        x1, y1, x2, y2 = win32gui.GetWindowRect(hwnd)
        print(x2 - x1, y2 - y1)
        pydirectinput.click(x1 +20, y1 + 20)
        pydirectinput.press("enter")



        while True :
            pydirectinput.press("enter")
            time.sleep(0.5)

            if (isEnd(x1, y1)) :
                break

            print("number : ", count)
            temp_data = getData(x1, y1)
            if (type(temp_data) == pd.Series) :
                data_list.append(temp_data)

                pydirectinput.press("esc")
                time.sleep(1)
                pydirectinput.press("up")
                count += 1
                time.sleep(1)
            else :
                break


        data_set = pd.DataFrame(data_list)
        data_set.columns = ['state', 'kind', 'sex', 'neutralization', 'color', 'birth', 'weight', 'number', 'period', 'location', 'feature', 'center', 'department']

        data_set.to_csv("train_data.csv", mode="w")