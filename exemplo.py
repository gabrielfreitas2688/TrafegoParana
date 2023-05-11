##Importantdo as bibliotecas
import cv2
import pytesseract

#Processamento inicial da imagem 
def RoiPlaca(source):
    ##Carrega a imagem
    imagem = cv2.imread("assets/exemplos/carro.jpg")
    cv2.imshow("teste imagem", imagem)

    ##Altera a imagem para escala cinza pois é possível apenas encontrar padrões utilizando preto e branco
    imagemcinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
   
    ##Converte todas as cores para preto e banco (binariza a imagem)
    _, bin = cv2.threshold(imagemcinza, 90, 255, cv2.THRESH_BINARY)
  
    ##Realzia o desfoque da imagem afim de melhorar a nitidez e encontrar os melhores contornos 
    desfoque = cv2.GaussianBlur(bin, (5,5), 0)

    ##Retorna os contornos fechados da imagem 
    contornos, hier = cv2.findContours(desfoque, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    ##Desenha os contornos na imagem principal(imagem, contornos, ID do contorno "-1" pega todos, cor, espeçura)
    cv2.drawContours(imagem, contornos, -1, (0, 255, 0), 2)
   # cv2.imshow("contornos", imagem)

    for c in contornos:
        ##Percorre os contornos fechados
        perimetro = cv2.arcLength(c, True)
        ##Considera apenas os retangulos maiores para destacar 
        if perimetro > 120:
            ##Aproxima o contorno à forma mais próxima dela mesma
            aprox = cv2.approxPolyDP(c, 0.03 * perimetro, True)
            ##Procura se a forma aproximada possui 4 lados, retangulo ou quadrados. (placas)
            if len(aprox) == 4:
                (x, y, alt, larg) = cv2.boundingRect(c)
                ##Passa a posição dos quadrados perfeitos e os pinta
                cv2.rectangle(imagem, (x, y), (x + alt, y + larg), (0, 255, 0), 2)
                #Encontra somente a parte pintada, considerando a posição + largura e altura
                corte = imagem[y:y + larg, x:x + alt]

                cv2.imwrite("assets/cortes/roi-img.jpg", corte)

    cv2.imshow("draw", imagem)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Pre processamento do recorte da placa
def preProcessamentoRoi():
    img_roi = cv2.imread("assets/cortes/roi-img.jpg")
    #Se não tiver placa ele não retorna nada
    if img_roi is None:
        return
    
    #Aumenta o tamanho da imagem para o tesseract não ter problema em ler 
    img_tamanho = cv2.resize(img_roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    ##Altera a imagem para escala cinza 
    img_roi_cinza = cv2.cvtColor(img_tamanho, cv2.COLOR_BGR2GRAY)

    ##Deixa a imagem binarizara (preto e branco)
    _, img_roi_bin = cv2.threshold(img_roi_cinza, 52, 255, cv2.THRESH_BINARY)

    cv2.imwrite("assets/cortes/roi-bin.jpg", img_roi_bin)

    cv2.imshow("roi", img_roi_bin)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Passando o tesseract na imagem 
def ocrImagemRoiPlaca():
    #Pega a imagem
    img_roi = cv2.imread("assets/cortes/roi-img.jpg")

    #configuração para ler apenas letras maiúsculas e numeros 
    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'

    #Saída (usado a biblitoca ingles pois PTBR tem erros)
    saida  = pytesseract.image_to_string(img_roi, lang="eng", config=config)

    print(saida)

if __name__ == "__main__":

    source = "assets/exemplos/carro.jpg";
    
    RoiPlaca(source)
    #preProcessamentoRoi()
    #ocrImagemRoiPlaca()
 