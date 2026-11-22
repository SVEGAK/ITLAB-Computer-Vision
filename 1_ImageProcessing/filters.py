import cv2
import numpy as np
import random
import math
rng = np.random.default_rng()
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
    img = img.astype(np.float64)
    factor = 0.8 * extent # Коэфф снижения контраста
    lift = 80 * extent  #поднятие черного
    M = np.array([#Используем афинное преобразование каждого пикселя B*factor + lift, G*factor + lift, R*factor + lift
    [factor, 0, 0],
    [0, factor, 0],
    [0, 0, factor]
], dtype=np.float32)
    res = cv2.transform(img,M) #Теперь домножаем каждый пиксель на матрицу преобразования
    res = cv2.add(res,lift)
    return np.clip(res,0.0,255.0).astype(np.uint8)
def calc_cage(x,y,img, direction):
        img_area = [img.shape[1]//250,img.shape[0]//250]
        if (direction[0] > 0):
            if (x + img_area[0] > img.shape[1]):
                x_right = img.shape[1] 
            else:
                x_right = x + img_area[0]
            x_left  = x
        else:
            if (x - img_area[0] <= 0):
                x_left = 0
            else:
                x_left = x - img_area[0]
            x_right = x
        if (direction[1]>0):
            if (y + img_area[1] > img.shape[0]):
                y_down = img.shape[0]
            else:
                y_down = y + img_area[1]
            y_upper = y
        else:
            if (y - img_area[1] <= 0):
                y_upper = 0
            else:
                y_upper = y - img_area[1]
            y_down = y

            
        result = [ [ x_left,x_right+1],[y_upper,y_down+1]]
        return result
    
def calc_rand_lines(x, y, img, count,direction,rng=rng):
    n = count*2
    pts_array = np.zeros((n,2)).astype(np.int32)
    shifts = rng.uniform(-1,2,size=(n,2))
    for i in range(0, n):
        if (i > 0):
            if (rng.integers(0, 2)):
                shifted_x = pts_array[i-1][0]-shifts[i][0]
                shifted_y = pts_array[i-1][1]-shifts[i][1]
                if (shifted_x < 0): shifted_x = 0
                if (shifted_x > x): shifted_x-=1
                if (shifted_y < 0): shifted_y = 0
                if (shifted_y > y): shifted_y-=1
                pts_array[i] = [shifted_x,shifted_y]
                continue
                
        if (i == 0):
            cages = [ [0,x],[0,y]]
        else:
            cages = calc_cage(pts_array[i-1][0],pts_array[i-1][1],img,direction)
        pts_array[i] = (rng.integers(cages[0][0],cages[0][1]),rng.integers(cages[1][0],cages[1][1]))
            
    return np.array(pts_array, dtype=np.int32).reshape((-1, 1, 2))
    
def old_photography(img, extent = 0.8, circles_count = 3, sharpes_count = 8):
    if (extent < 0):
        print("Extent is lower than zero.")
        return img
    elif (extent == 0):
        return img
    if (sharpes_count < 0):
        print("Sharpes count is lower than zero.")
        return img
    if (circles_count < 0):
        print("Circles count is lower than zero.")
        return img
    res = fade_color(img,0.8).astype(np.int16)
    
    #Шум
    noise_matrix = np.random.randint( int(-30*extent), int(30*extent),res.shape,dtype=np.int16)
    res = cv2.add(res,noise_matrix) #В аргументах преобразуем исходную в signet int, чтобы избежать переполнения
    res = np.clip(res,0,255).astype(np.float64)
    
    #царапины и пятна
    sharp_mask = np.zeros(res.shape,dtype=np.float64)
    for i in range(0,circles_count):
        sharp_mask = cv2.circle(sharp_mask,(random.randint(0,res.shape[1]+1),random.randint(0,res.shape[0]+1)),random.randint(0,res.shape[0]//4),(1,9,14),-1)
    x = res.shape[1]
    y = res.shape[0]
    new_direction = rng.integers(-1,2,(2,2))
    for s in range(sharpes_count):
        if (s%2): direction = new_direction[1]
        else: direction = new_direction[0]
        pts = calc_rand_lines(x, y, img, rng.integers(15,500), direction) #direction - int кортеж -> направление линии по x и y
        for j in range(len(pts) - 1):                       # 0..4 — все 5 отрезков
            sharp_mask = cv2.line(
                sharp_mask,
                tuple(pts[j][0]),                           # (x, y) начала
                tuple(pts[j+1][0]),                         # (x, y) конца
                (128, 128, 128),
                1,
                lineType=cv2.LINE_8,
            )
    
    sharp_mask = cv2.GaussianBlur(sharp_mask, (5,5),0)
    res = np.clip(cv2.add(res,sharp_mask),0.0,255.0)
    
    return res.astype(np.uint8)

def antique_filter(img,extent=0.9,circles_count = 1,sharpes_count = 7, vinyet_scale = 0.5):
    if (extent < 0):
        print("Extent is lower than zero.")
        return img
    elif (extent == 0):
        return img
    if (sharpes_count < 0):
        print("Sharpes count is lower than zero.")
        return img
    if (circles_count < 0):
        print("Circles count is lower than zero.")
        return img
    if (vinyet_scale < 0):
        print("Vinyet scale is lower than zero.")
        return img
    x = img.shape[1]
    y = img.shape[0]
    res = cv2.pyrDown(img)
    res = cv2.pyrUp(res) #Немного размываем изображение
    
    #Шум + царапины
    res = old_photography(res,extent,circles_count,sharpes_count)
    
    
    #Эффект сепии
    res = convert_to_grayscale(res)
    bgr_matrix = np.array([[0.20],[0.75],[1.07]])
    res = cv2.transform(res, bgr_matrix) #Возвращаем обратно в BGR
    
    res= res.astype(np.float64) #float64 нужен из за дальнейшего сложения
    if (x > y):
        vinyet_radius = int(vinyet_scale*y)
    else: vinyet_radius = int(vinyet_scale*x)
    vinyet_mask = cv2.circle(np.zeros(res.shape),(res.shape[1]//2,res.shape[0]//2),vinyet_radius,(255,255,255),-1) #Черный экран с белым кругом по центру (маска для виньетки)
    vinyet_mask = cv2.GaussianBlur(vinyet_mask, calc_ksize(res),0) #Размываем круг чтобы сгладить его углы
    vinyet_mask = np.clip(vinyet_mask,0.0,255.0)/255.0# Приводим к значением от 0.0 до 1.0
    res *= vinyet_mask
    
    res = res.astype(np.uint8)
    return res
 

        
def mask(img,horizontal = 0.5, vertical = 0.5):
    res = img
    x = img.shape[1]
    y = img.shape[0]
    vertical_r = int(vertical*y)
    horizontal_r = int(horizontal*x)
    mask = np.full(res.shape,255,dtype=np.uint8)
    mask = cv2.ellipse(mask,(res.shape[1]//2,res.shape[0]//2),(horizontal_r,vertical_r),0,0,360,(0,0,0),-1)
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

def calc_gauss_kernel(extent,base):
    result = math.floor(base*extent)
    if (result < 3): return (3,3)
    if(result%2 == 0):
        result +=1
    return (result,result)

def neon_effect(img,B,G,R, extent,glow): # IMG B G R extent(float/int) extent - фактически четкость границ неона - чем меньше тем больше границ выделяются неоном
    if not 0 <= extent <= 1:
        raise ValueError("extent must be in [0, 1]")
    if not 0 <= glow <= 10:
        raise ValueError("extent must be in [0, 10]")
    if extent == 0:
        return img.copy()
    
    neon = img.astype(np.float64)
    gaussed = cv2.GaussianBlur(neon, calc_gauss_kernel(extent,9), 8.5)
    gray = convert_to_grayscale(gaussed)#Чтобы потом удобнее задать неоновый цвет
    
    sharrx = cv2.Scharr(gray,cv2.CV_64F,1,0)
    sharry = cv2.Scharr(gray,cv2.CV_64F,0,1)
    
    sharr = np.sqrt((sharrx*sharrx+sharry*sharry))
    
    sharr = cv2.threshold(np.clip(sharr,0.0,255.0), 220*extent, 255, cv2.THRESH_TOZERO)[1]
    if (extent < 0.75):
        laplacian = cv2.Laplacian(cv2.GaussianBlur(gray, calc_gauss_kernel(extent,15),0),cv2.CV_64F,dst=None,ksize=7)
        laplacian = np.clip(cv2.convertScaleAbs(laplacian*pow(6,extent)),0.0,255.0)
        laplacian = (cv2.threshold(laplacian, 50*extent, 255, cv2.THRESH_BINARY))[1]#Пороговая обработка, чтобы оставить самые яркие участки(контуры)
        neon_mask = np.clip(np.sqrt(sharr*laplacian),0.0,255.0).astype(np.uint8)
        
    else: neon_mask = np.clip(sharr,0.0,255.0).astype(np.uint8)
    bgr_matrix = np.array([[B],[G],[R]])#Матрица цвета
    bgr_neon_mask = cv2.transform(neon_mask, bgr_matrix) #Возвращаем обратно в BGR
    
    glow_radius = glow / 10.0
    sigma = 1.0 + 8.0 * glow_radius
    radius = max(1, round(3 * sigma))
    ksize = 2 * radius + 1

    neon_blurred = cv2.GaussianBlur(bgr_neon_mask, (ksize,ksize),sigma) #Размываем копию чтобы получить свечение

    neon = cv2.add(bgr_neon_mask,neon_blurred) #складываем и получаем свечение на изначальную маску
    img_dark = (img*0.6).astype(np.uint8)#Затемняем изображение, чтобы на ярких картинках не было переполнения
    res = cv2.add(img_dark,neon)#Накладываем неоновый эффект
    return res
