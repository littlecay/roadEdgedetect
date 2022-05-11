import cv2
import numpy as np

cap = cv2.VideoCapture('82698e5874ffb94ad2e8a338d0bd14bf.mp4')
fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fcnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
R = 0
G = 0
B = 0

for _frame in range(fcnt):
    ret, frame = cap.read()
    B = frame[0].sum()/(width*height)
    G = frame[1].sum()/(width*height)
    R = frame[2].sum()/(width*height)


