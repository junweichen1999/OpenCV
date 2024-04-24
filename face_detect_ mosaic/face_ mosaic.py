import cv2

""".py檔案讀取(模型)失敗，但是jypiternotebook正常"""

# 取得影片之路徑 
# cap = cv2.VideoCapture("video_test01.mp4")
# 取得視訊鏡頭的畫面
cap = cv2.VideoCapture(0)
screenshot_count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=1.2, fy=1.2)

    if ret:
        # 轉灰階
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 選定模型
        faceCascade = cv2.CascadeClassifier(r"haarcascade_frontalcatface.xml")
        # 檢查是否成功加載
        if faceCascade.empty():
            print("Error: Unable to load cascade classifier file.")
        else:
            print("Cascade classifier file loaded successfully.")
        # 用灰階的套模 (勾勒出輪廓回傳位子)
        face_rect = faceCascade.detectMultiScale(gray, 1.1, 8)
        count_how_many_people = len(face_rect)
        for (x, y, w, h) in face_rect:
            mosaic = frame[y:y+h, x:x+w]
            level = 15
            mh = int(h/level)
            mw = int(w/level)
            mosaic = cv2.resize(mosaic, (mw,mh), interpolation=cv2.INTER_LINEAR)
            mosaic = cv2.resize(mosaic, (w,h), interpolation=cv2.INTER_NEAREST)
            frame[y:y+h, x:x+w] = mosaic
        cv2.imshow('oxxostudio', frame)
       
    else:
        break
    # 每取得1針 會等待100毫秒 如果監聽到q按下去(跳出迴圈)waitKey等待鍵盤某個鍵被按下去
    if cv2.waitKey(10) == ord("q"):
        cv2.destroyAllWindows()
        break

# 釋放攝像頭資源
cap.release()
cv2.destroyAllWindows()
