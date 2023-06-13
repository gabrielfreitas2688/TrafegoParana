import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
from sort import *

video = cv2.VideoCapture("video3.mp4")

model = YOLO("yolov8n.pt")

tracker = Sort(max_age=20)

classNames = ["pessoa", "bicicleta", "carro", "moto", "avião", "ônibus", "trem", "caminhão", "barco",
              "semáforo", "hidrante", "placa de pare", "parquímetro", "banco", "pássaro", "gato",
              "cachorro", "cavalo", "ovelha", "vaca", "elefante", "urso", "zebra", "girafa", "mochila", "guarda-chuva",
              "bolsa", "gravata","mala", "frisbee", "esquis", "snowboard", "bola esportiva", "pipa", "taco de beisebol",
              "luva de beisebol", "skate", "prancha de surf", "raquete de tênis", "garrafa", "copo de vinho", "taça",
              "garfo", "faca", "colher", "tigela", "banana", "maçã", "sanduíche", "laranja", "brócolis",
              "cenoura", "cachorro-quente", "pizza", "rosquinha","bolo", "cadeira", "sofá", "vaso de planta", "cama",
              "mesa de jantar", "banheiro", "monitor de tv", "laptop", "mouse", "controle remoto", "teclado", "telefone celular",
              "microondas", "forno", "torradeira", "pia", "geladeira", "livro", "relógio", "vaso", "tesoura",
              "ursinho de pelúcia", "secador de cabelo", "escova de dentes"
              ]
mask = cv2.imread('mask3.png')
linha = [1,260,608,260]
contador = []

while True:
    _,img = video.read()
    img = cv2.resize(img,(608,342))
    imgRegion = cv2.bitwise_and(img,mask)

    results = model(imgRegion,stream=True)
    detections = np.empty((0,5))


    for obj in results:
        dados = obj.boxes
        for x in dados:
            #bounding boxes
            x1,y1,x2,y2 = x.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1, y2-y1
            cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(255,0,0),3)
            #conf
            conf = int(x.conf[0]*100)
            #classe
            cls = int(x.cls[0])
            nomeClass = classNames[cls]
            if conf >=20 and nomeClass == "carro" or nomeClass=="caminhão" or nomeClass=="moto":
                # cvzone.cornerRect(img,(x1,y1,w,h),colorR=(255,0,255))
                # cvzone.putTextRect(img,nomeClass,(x1,y1-10),scale=1.5,thickness=2)
                crArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,crArray))

    resultTracker = tracker.update(detections)

    cv2.line(img,(linha[0],linha[1]),(linha[2],linha[3]),(0,0,255),5)

    for result in resultTracker:
        x1,y1,x2,y2,id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255, 0, 255))
        cvzone.putTextRect(img, str(int(id)), (x1, y1 - 10), scale=1, thickness=1)
        cx,cy = x1+w//2, y1+h//2
        cv2.circle(img,(cx,cy),2,(255,0,255),2)

        if linha[0] <cx < linha[2] and linha[1] -15 <cy <linha[1] + 15:
            if contador.count(id) ==0:
                contador.append(id)
                cv2.line(img, (linha[0], linha[1]), (linha[2], linha[3]), (0, 255, 0), thickness=1)

    #print(len(contador))
    cvzone.putTextRect(img, "Veiculos: " + str(len(contador)), (20, 70), scale=1.5, thickness=2)

    cv2.imshow('IMG',img)
    cv2.waitKey(1)