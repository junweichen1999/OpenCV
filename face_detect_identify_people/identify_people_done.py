import cv2
recognizer = cv2.face.LBPHFaceRecognizer_create()         # 啟用訓練人臉模型方法
recognizer.read('face.yml')                               # 讀取人臉模型檔
cascade_path = "detect_frontface.xml"                     # 載入人臉追蹤模型
face_cascade = cv2.CascadeClassifier(cascade_path)        # 啟用人臉追蹤

#建立姓名ID對照之人臉
name = {
    "1":"joeman",
    "2":"lovesand",
    "3":"hook",
    "4":"wilson",
}
# 啟動鏡頭
cap = cv2.VideoCapture(0)
screenshot_count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
    if ret:
        # 轉灰階
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        # 識別人臉
        faces = face_cascade.detectMultiScale(gray)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)            # 標記人臉外框
        idnum,confidence = recognizer.predict(gray[y:y+h,x:x+w])  # 取出 id 號碼以及信心指數 confidence
        if confidence > 85:
            text = name[str(idnum)]                               # 如果信心指數小於 60，取得對應的名字
        else:
            text = '???'                                          # 不然名字就是 ???
        # 在人臉外框旁加上名字
        cv2.putText(frame, text, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow('identify_peoples_face', frame)

    key = cv2.waitKey(5)
    if key == ord("s"):
        # Increment the screenshot count
        screenshot_count += 1
        # Save the current frame as an image
        cv2.imwrite(f"screenshot(confidence_greater_than_85)_{screenshot_count}.jpg", frame)
        print(f"Screenshot saved as screenshot_{screenshot_count}.jpg")

    if cv2.waitKey(5) == ord('q'):
        break    # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()

       
