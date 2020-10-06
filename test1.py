import os
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
    img = cv2.resize(imgGray,(64,64))

    # リシェイプ
    img = np.reshape(img,(1,64,64))

    # Torch化
    img = torch.from_numpy(img.astype(np.float32)).clone()

    #推論
    output = model(img)

    #予測値
    p = model.forward(img)

    #予測値出力
    print(p)

    print(p.argmax())

    print(type(p))


if __name__ == "__main__":
    model = yutokodama.Net(num=6)
    PATH = "nn.pt"
    model.load_state_dict(torch.load(PATH))
    model.eval()

    # テスト
    andou = cv2.imread("Resources/train/suetomo/1.jpg")

    f(andou)

    