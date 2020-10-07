import os
import sys
import cv2
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import nn as yutokodama

def f(imgCV):
    # チャンネル数を１
    imgGray = cv2.cvtColor(imgCV,cv2.COLOR_BGR2GRAY)

    #リサイズ
    img = cv2.resize(imgGray,(128,128))

    # リシェイプ
    img = np.reshape(img,(1,128,128))

    # transpose h,c,w
    img = np.transpose(img,(1,2,0))

    # ToTensor 正規化される
    img = img.astype(np.uint8)
    mInput = transforms.ToTensor()(img)  
    #print(mInput)

    #推論
    #print(mInput.size())
    output = model(mInput[0])

    #予測値
    p = model.forward(mInput)

    #予測値出力
    print(p)
    print(p.argmax())
    print(type(p))
    if p.argmax() == 0:
        sys.stdout.write("\r {:<1s} さんです".format("安藤"))
    if p.argmax() == 1:
        sys.stdout.write("\r {:<1s} さんです".format("東"))
    if p.argmax() == 2:
        sys.stdout.write("\r {:<1s} さんです".format("片岡"))
    if p.argmax() == 3:
        sys.stdout.write("\r {:<1s} さんです".format("小玉"))
    if p.argmax() == 4:
        sys.stdout.write("\r {:<1s} さんです".format("増田"))
    if p.argmax() == 5:
        sys.stdout.write("\r {:<1s} さんです".format("末友"))

if __name__ == "__main__":
    model = yutokodama.Net(num=6)
    PATH = "nn.pt"
    model.load_state_dict(torch.load(PATH))
    model.eval()

    cascadePath = "haarcascades/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascadePath)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920/10)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080/10)
    color = (255,0,255)
    while True:
        success,img = cap.read()
        imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgResults = img.copy()
        #cascade = cv2.CascadeClassifier(cascadePath)
        facerect = cascade.detectMultiScale(imgGray,scaleFactor=1.1,minNeighbors=2,minSize=(10,10))
        color = (255,0,255)
        if len(facerect) > 0:
            for (x,y,w,h) in facerect:
                cv2.rectangle(imgResults,(x,y),(x+w,y+h),color,thickness=2)
                imgTrim = img[y:y+h,x:x+w]
                cv2.imshow("Trim",imgTrim)
                cv2.imshow("Results",imgResults)
                cv2.imwrite("hoge.jpg",imgTrim)
                imgTrim = cv2.imread("hoge.jpg")
                # 推論
                f(imgTrim)
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    