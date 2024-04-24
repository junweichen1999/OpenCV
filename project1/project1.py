import cv2
# 取得影片之路徑 
#cap = cv2.VideoCapture("video_test01.mp4")
# 取得視訊鏡頭的畫面
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    if ret:
        cv2.imshow("video", frame)
    else:
        break
    #每取得1針 會等待100毫秒 如果監聽到q按下去(跳出迴圈)waitKey等待鍵盤某個鍵被按下去
    if cv2.waitKey(1) == ord("q"):
        #cv2.destroyAllWindows() 
        break