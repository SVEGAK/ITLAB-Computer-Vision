import cv2
import numpy as np
img1  = cv2.imread("cat.jpg")
img2 = cv2.imread('cat2.png')

def resize_img(img,new_w,new_h):
    k_x = new_w/img.shape[1] #Ширина
    k_y = new_h/img.shape[0] #Высота
    #Строим матрицу преобразования
    M = np.array([[k_x,0,0],
                  [0,k_y,0]])
    return cv2.warpAffine(img, M, (new_w,new_h))

def convert_to_grayscale(img): #Через матричное умножение, 
    if (img.shape)[2] == 1:
        return img
    grey_matrix = np.array([0.114,0.587,0.299]) #Матрица преобразования пикселя в серый цвет
    #(фактически серый цвет в BGR), то есть BGR схлопываем до числа [0,255] яркости пикселя
    return cv2.transform(img, grey_matrix) #Умножение каждого пикселя на матрицу преобразования - теперь каждый пиксель это число показатель яркости
                                                
def antique_filter(img):
    #Сначала сделаем изображение полутоновым
    res = convert_to_grayscale(img)
    bgr_matrix = np.array([[0.20],[0.75],[1.07]])
    res = cv2.transform(res, bgr_matrix) #Возвращаем обратно в BGR и получаем эффект сепии
    res = cv2.pyrDown(res)
    res = cv2.pyrUp(res) #Немного размываем изображение
    
    return res
#img1 = resize_img(img1,500,500)
img2 = antique_filter(img2)
img1 = convert_to_grayscale(img1)
if img1 is None:
    print("Ничего не получилось")
else:
    cv2.imshow('Image',img1)
    cv2.imshow('Image2',img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
