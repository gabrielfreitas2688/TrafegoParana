import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
from sort import *

video = cv2.VideoCapture("video3.mp4")

model = YOLO("yolov8n.pt")


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
while True:
    _,img = video.read()
    img = cv2.resize(img,(608,342))
    results = model(img,stream=True)


    for obj in results:
        dados = obj.boxes
        for x in dados:
            #bounding boxes
            x1,y1,x2,y2 = x.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1, y2-y1
            cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(255,0,0),3)
                # cvzone.cornerRect(img,(x1,y1,w,h),colorR=(255,0,255))
                # cvzone.putTextRect(img,nomeClass,(x1,y1-10),scale=1.5,thickness=2)


    cv2.imshow('IMG',img)
    cv2.waitKey(1)