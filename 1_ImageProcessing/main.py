import cv2
import numpy as np
import random
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
                                                
def calc_ksize(img):
    h, w = img.shape[:2]
    ksize = int(min(h, w)/2)
    if ksize % 2 == 0:
        ksize += 1 #ksize - нечетное число
    return (ksize,ksize)


def fade_color(img,extent = 1):
    if (extent < 0):
        print("Extent is lower than zero.")
        return img
    elif (extent == 0):
        return img
    factor = 0.8 * extent # Коэфф снижения контраста
    lift = 80 * extent  #поднятие черного
    M = np.array([#Используем афинное преобразование каждого пикселя B*factor + lift, G*factor + lift, R*factor + lift
    [factor, 0, 0],
    [0, factor, 0],
    [0, 0, factor]
], dtype=np.float32)
    res = cv2.transform(img,M) #Теперь домножаем каждый пиксель на матрицу преобразования
    res = cv2.add(res,lift)
    return res

def calc_rand_lines(x, y, img):
    if (x - (img.shape[1]//4) <= 0):
        x_left = 0
    else:
        x_left = x - img.shape[1]//4
        
    if (y - (img.shape[1]//4) <= 0):
        y_left = 0
    else:
        y_left = y - img.shape[0]//4 
    
    if (x + (img.shape[1]//4) > img.shape[1]):
        x_right = img.shape[1] 
    else:
        x_right = x + img.shape[1]//4 
        
    if (y + (img.shape[1]//4) > img.shape[0]):
        y_right = img.shape[0]
    else:
        y_right = y + img.shape[0]//4 
    return (random.randint(x_left,x_right+1),random.randint(y_left,y_right+1))
    
def old_photography(img, extent = 0.8):
    res = fade_color(img.astype(np.float64),0.805)
    
    #Шум
    noise_matrix = np.random.randint( int(-30*extent), int(30*extent),res.shape,dtype=np.int16)
    res = cv2.add(res.astype(np.int16),noise_matrix) #В аргументах преобразуем исходную в signet int, чтобы избежать переполнения
    res = np.clip(res,0,255).astype(np.float64)
    
    #царапины и пятна
    sharp_mask = np.zeros(res.shape,dtype=np.float64)
    for i in range(0, int(extent*4)):
        x = res.shape[1]
        y = res.shape[0]
        a = (random.randint(0,x),random.randint(0,y))
        b = (calc_rand_lines(a[0], a[1], img))
        c = calc_rand_lines(b[0], b[1], img)
        
        pts = np.array([a, b, c], dtype=np.int32).reshape((-1, 1, 2))
        
        sharp_mask = cv2.polylines(sharp_mask, [pts], False, (255, 255, 255), 1)

    for i in range(0,int(3*extent)):
        sharp_mask = cv2.circle(sharp_mask,(random.randint(0,res.shape[1]),random.randint(0,res.shape[0])),random.randint(0,res.shape[0]//4),(1,12,17),-1)
        
    sharp_mask = cv2.GaussianBlur(sharp_mask,(31,31),3.2,3.2)
    res = np.clip(cv2.add(res,sharp_mask),0.0,255.0)
    
    return res.astype(np.uint8)

def antique_filter(img):
#Сначала сделаем изображение полутоновым
     res = img
     res = cv2.pyrDown(res)
     res = cv2.pyrUp(res) #Немного размываем изображение
     
     #Шум + царапины
     res = old_photography(res,1)
     
     
     #Эффект сепии
     res = convert_to_grayscale(res)
     bgr_matrix = np.array([[0.20],[0.75],[1.07]])
     res = cv2.transform(res, bgr_matrix) #Возвращаем обратно в BGR
     
     res= res.astype(np.float64) #float64 нужен из за дальнейшего сложения
     
     
     vinyet_mask = cv2.circle(np.zeros(res.shape),(res.shape[1]//2,res.shape[0]//2),res.shape[0]//2,(255,255,255),-1) #Черный экран с белым кругом по центру (маска для виньетки)
     vinyet_mask = cv2.GaussianBlur(vinyet_mask, calc_ksize(res),0) #Размываем круг чтобы сгладить его углы
     vinyet_mask = np.clip(vinyet_mask,0.0,255.0)/255.0# Приводим к значением от 0.0 до 1.0
     res *= vinyet_mask
     
     res = res.astype(np.uint8)
     return res
 

        
def mask(img):
    res = img
    mask = np.full(res.shape,255,dtype=np.uint8)
    mask = cv2.ellipse(mask,(res.shape[1]//2,res.shape[0]//2),(res.shape[1]//2,res.shape[0]//2),0,0,360,(0,0,0),-1)
    mask = cv2.GaussianBlur(mask, (109,109),0) #Размываем круг чтобы сгладить его углы
    res = cv2.add(res,mask)
    res = np.clip(res,0.0,255.0)
    res = res.astype(np.uint8)
    return res

def negative_film(img):
    #Негативный фильтр - инвертирование цветов, т.е. цвет = 255 - старый_цвет
    res = img
    mask = np.full(res.shape,255,dtype=np.uint8)
    res = cv2.subtract(mask, res)
    return res.astype(np.uint8)



img2_res = old_photography(img2)
img1_res = negative_film(img1)
if img1 is None:
    print("Ничего не получилось")
else:
    cv2.imshow('Image3',img2_res)
    cv2.imshow('Image1',img1_res)
    cv2.imshow('Image123',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
