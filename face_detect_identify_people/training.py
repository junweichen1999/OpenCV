import cv2
import numpy as np

#與jypiter相同但是讀取檔案都有問題

detector = cv2.CascadeClassifier('detect_frontface.xml')  # 載入人臉追蹤模型
recog = cv2.face.LBPHFaceRecognizer_create()      # 啟用訓練人臉模型方法
faces = []   # 儲存人臉位置大小的串列
ids = []     # 記錄該人臉 id 的串列

def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"無法讀取影像檔案：{path}")
    return img

try:
    for i in range(1, 31):
        img = read_image(f'./face01/{i}.jpg')          #訓練joman
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_np = np.array(gray, 'uint8')
        face = detector.detectMultiScale(gray)
        for (x, y, w, h) in face:
            faces.append(img_np[y:y + h, x:x + w])
            ids.append(1)

    for i in range(1,32):
        img = cv2.imread(f'./face02/{i}.jpg')         # 訓練艾利沙沙 (圖片應該有損毀)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 
        img_np = np.array(gray,'uint8')               # 
        face = detector.detectMultiScale(gray)        # 
        for(x,y,w,h) in face:
            faces.append(img_np[y:y+h,x:x+w])         
            ids.append(2)    


    for i in range(1,32):
        img = cv2.imread(f'./face03/{i}.jpg')           # 訓練hook的臉
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        img_np = np.array(gray,'uint8')               
        face = detector.detectMultiScale(gray)        
        for(x,y,w,h) in face:
            faces.append(img_np[y:y+h,x:x+w])         
            ids.append(3)                            

    for i in range(1,31):                               # 訓練自己的臉部
        img = cv2.imread(f'./face04/{i}.jpg')           
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        img_np = np.array(gray,'uint8')               
        face = detector.detectMultiScale(gray)        
        for(x,y,w,h) in face:
            faces.append(img_np[y:y+h,x:x+w])         
            ids.append(4)                             

    # Repeat the process for other directories and IDs...

    print('訓練中...')
    recog.train(faces, np.array(ids))
    recog.save('face.yml')
    print('訓練完成！')
except Exception as e:
    print(f"發生錯誤：{e}")
