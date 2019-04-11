import pandas as pd
import cv2
import numpy as np
from Lista02 import FuncoesML as fun
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import time
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

#lendo a imagem
print('comecou load images')
squares = []
squares = fun.loadFiles('C:/Users/Auricelia/Desktop/DataSetsML/New_shapes_dataset/squares/*.jpg', squares)

circles = []
circles = fun.loadFiles('C:/Users/Auricelia/Desktop/DataSetsML/New_shapes_dataset/circles/*.jpg',circles)

triangles = []
triangles = fun.loadFiles('C:/Users/Auricelia/Desktop/DataSetsML/New_shapes_dataset/triangles/*.jpg', triangles)

ellipses = []
ellipses = fun.loadFiles('C:/Users/Auricelia/Desktop/DataSetsML/New_shapes_dataset/ellipses/*.jpg', ellipses)

trapezia = []
trapezia = fun.loadFiles('C:/Users/Auricelia/Desktop/DataSetsML/New_shapes_dataset/trapezia/*.jpg', trapezia)

rectangles = []
rectangles = fun.loadFiles('C:/Users/Auricelia/Desktop/DataSetsML/New_shapes_dataset/rectangles/*.jpg', rectangles)

rhombuses = []
rhombuses = fun.loadFiles('C:/Users/Auricelia/Desktop/DataSetsML/New_shapes_dataset/rhombuses/*.jpg', rhombuses)

lines = []
lines = fun.loadFiles('C:/Users/Auricelia/Desktop/DataSetsML/New_shapes_dataset/lines/*.jpg', lines)

hexagons = []
hexagons = fun.loadFiles('C:/Users/Auricelia/Desktop/DataSetsML/New_shapes_dataset/hexagons/*.jpg', hexagons)

print('terminou load images')

# Selecionando aleatoriamente 72 imagens de cada classe

squares_selec, squares_naoselec = fun.seleciona_imagens(squares,72)
circles_selec, circles_naoselec = fun.seleciona_imagens(circles,72)
triangles_selec, triangles_naoselec = fun.seleciona_imagens(triangles,72)
ellipses_selec, ellipses_naoselec = fun.seleciona_imagens(ellipses,72)
trapezia_selec, trapezia_naoselec = fun.seleciona_imagens(trapezia,72)
rectangles_selec, rectangles_naoselec = fun.seleciona_imagens(rectangles,72)
rhombuses_selec, rhombuses_naoselec = fun.seleciona_imagens(rhombuses,72)
lines_selec, lines_naoselec = fun.seleciona_imagens(lines,72)
hexagons_selec, hexagons_naoselec = fun.seleciona_imagens(hexagons,72)

#Salvando em pastas diferentes as imagens para seleção de características e as de teste

fun.save_images(squares_selec,'C:/Users/Auricelia/Desktop/DataSetsML/Random/RandomSquares/')
fun.save_images(squares_naoselec,'C:/Users/Auricelia/Desktop/DataSetsML/Not_Random/Squares/')

fun.save_images(circles_selec,'C:/Users/Auricelia/Desktop/DataSetsML/Random/RandomCircles/')
fun.save_images(circles_naoselec,'C:/Users/Auricelia/Desktop/DataSetsML/Not_Random/Circles/')

fun.save_images(triangles_selec,'C:/Users/Auricelia/Desktop/DataSetsML/Random/RandomTriangles/')
fun.save_images(triangles_naoselec,'C:/Users/Auricelia/Desktop/DataSetsML/Not_Random/Triangles/')

fun.save_images(ellipses_selec, 'C:/Users/Auricelia/Desktop/DataSetsML/Random/RandomEllipses/')
fun.save_images(ellipses_naoselec,'C:/Users/Auricelia/Desktop/DataSetsML/Not_Random/Ellipses/')

fun.save_images(trapezia_selec,'C:/Users/Auricelia/Desktop/DataSetsML/Random/RandomTrapezia/')
fun.save_images(trapezia_naoselec,'C:/Users/Auricelia/Desktop/DataSetsML/Not_Random/Trapezia/')

fun.save_images(rectangles_selec,'C:/Users/Auricelia/Desktop/DataSetsML/Random/RandomRectangles/')
fun.save_images(rectangles_naoselec, 'C:/Users/Auricelia/Desktop/DataSetsML/Not_Random/Rectangles/')

fun.save_images(rhombuses_selec,'C:/Users/Auricelia/Desktop/DataSetsML/Random/RandomRhombuses/')
fun.save_images(rhombuses_naoselec, 'C:/Users/Auricelia/Desktop/DataSetsML/Not_Random/Rhombuses/')

fun.save_images(lines_selec, 'C:/Users/Auricelia/Desktop/DataSetsML/Random/RandomLines/')
fun.save_images(lines_naoselec, 'C:/Users/Auricelia/Desktop/DataSetsML/Not_Random/Lines/')

fun.save_images(hexagons_selec, 'C:/Users/Auricelia/Desktop/DataSetsML/Random/RandomHexagons/')
fun.save_images(hexagons_naoselec, 'C:/Users/Auricelia/Desktop/DataSetsML/Not_Random/Hexagons/')


# PRE PROCESSING

#criando cópias de cada uma das pastas para redimensionar as imagens
#quadrados

squares16_s = squares_selec.copy()
squares16_n = squares_naoselec.copy()

squares32_s = squares_selec.copy()
squares32_n = squares_naoselec.copy()

squares64_s = squares_selec.copy()
squares64_n = squares_naoselec.copy()

squares128_s = squares_selec.copy()
squares128_n = squares_naoselec.copy()

squares16_s = fun.resizeImages(squares16_s,16,16)
squares16_n = fun.resizeImages(squares16_n,16,16)

squares32_s = fun.resizeImages(squares32_s,32,32)
squares32_n = fun.resizeImages(squares32_n,32,32)

squares64_s = fun.resizeImages(squares64_s,64,64)
squares64_n = fun.resizeImages(squares64_n,64,64)

squares128_s = fun.resizeImages(squares128_s,128,128)
squares128_n = fun.resizeImages(squares128_n,128,128)

#círculos
circles16_s = circles_selec.copy()
circles16_n = circles_naoselec.copy()

circles32_s = circles_selec.copy()
circles32_n = circles_naoselec.copy()

circles64_s = circles_selec.copy()
circles64_n = circles_naoselec.copy()

circles128_s = circles_selec.copy()
circles128_n = circles_naoselec.copy()

circles16_s = fun.resizeImages(circles16_s,16,16)
circles16_n = fun.resizeImages(circles16_n,16,16)

circles32_s = fun.resizeImages(circles32_s,32,32)
circles32_n = fun.resizeImages(circles32_n,32,32)

circles64_s = fun.resizeImages(circles64_s,64,64)
circles64_n = fun.resizeImages(circles64_n,64,64)
circles128_s = fun.resizeImages(circles128_s,128,128)
circles128_n = fun.resizeImages(circles128_n,128,128)

#elipses
ellipsis16_s = ellipses_selec.copy()
ellipsis16_n = ellipses_naoselec.copy()

ellipsis32_s = ellipses_selec.copy()
ellipsis32_n = ellipses_naoselec.copy()

ellipsis64_s = ellipses_selec.copy()
ellipsis64_n = ellipses_naoselec.copy()

ellipsis128_s = ellipses_selec.copy()
ellipsis128_n = ellipses_naoselec.copy()

ellipsis16_s = fun.resizeImages(ellipsis16_s,16,16)
ellipsis16_n = fun.resizeImages(ellipsis16_n,16,16)

ellipsis32_s = fun.resizeImages(ellipsis32_s,32,32)
ellipsis32_n = fun.resizeImages(ellipsis32_n,32,32)

ellipsis64_s = fun.resizeImages(ellipsis64_s,64,64)
ellipsis64_n = fun.resizeImages(ellipsis64_n,64,64)

ellipsis128_s = fun.resizeImages(ellipsis128_s,128,128)
ellipsis128_n = fun.resizeImages(ellipsis128_n,128,128)

#hexágonos
hexagons16_s = hexagons_selec.copy()
hexagons16_n = hexagons_naoselec.copy()
hexagons32_s = hexagons_selec.copy()
hexagons32_n = hexagons_naoselec.copy()
hexagons64_s = hexagons_selec.copy()
hexagons64_n = hexagons_naoselec.copy()
hexagons128_s = hexagons_selec.copy()
hexagons128_n = hexagons_naoselec.copy()

hexagons16_s = fun.resizeImages(hexagons16_s,16,16)
hexagons16_n = fun.resizeImages(hexagons16_n,16,16)
hexagons32_s = fun.resizeImages(hexagons32_s,32,32)
hexagons32_n = fun.resizeImages(hexagons32_n,32,32)
hexagons64_s = fun.resizeImages(hexagons64_s,64,64)
hexagons64_n = fun.resizeImages(hexagons64_n,64,64)
hexagons128_s = fun.resizeImages(hexagons128_s,128,128)
hexagons128_n = fun.resizeImages(hexagons128_n,128,128)

#linhas
lines16_s = lines_selec.copy()
lines16_n = lines_naoselec.copy()
lines32_s = lines_selec.copy()
lines32_n = lines_naoselec.copy()
lines64_s = lines_selec.copy()
lines64_n = lines_naoselec.copy()
lines128_s = lines_selec.copy()
lines128_n = lines_naoselec.copy()

lines16_s = fun.resizeImages(lines16_s,16,16)
lines16_n = fun.resizeImages(lines16_n,16,16)
lines32_s = fun.resizeImages(lines32_s,32,32)
lines32_n = fun.resizeImages(lines32_n,32,32)
lines64_s = fun.resizeImages(lines64_s,64,64)
lines64_n = fun.resizeImages(lines64_n,64,64)
lines128_s = fun.resizeImages(lines128_s,128,128)
lines128_n = fun.resizeImages(lines128_n,128,128)

#retângulos
rectangles16_s = rectangles_selec.copy()
rectangles16_n = rectangles_naoselec.copy()
rectangles32_s = rectangles_selec.copy()
rectangles32_n = rectangles_naoselec.copy()
rectangles64_s = rectangles_selec.copy()
rectangles64_n = rectangles_naoselec.copy()
rectangles128_s = rectangles_selec.copy()
rectangles128_n = rectangles_naoselec.copy()

rectangles16_s = fun.resizeImages(rectangles16_s,16,16)
rectangles16_n = fun.resizeImages(rectangles16_n,16,16)
rectangles32_s = fun.resizeImages(rectangles32_s,32,32)
rectangles32_n = fun.resizeImages(rectangles32_n,32,32)
rectangles64_s = fun.resizeImages(rectangles64_s,64,64)
rectangles64_n = fun.resizeImages(rectangles64_n,64,64)
rectangles128_s = fun.resizeImages(rectangles128_s,128,128)
rectangles128_n = fun.resizeImages(rectangles128_n,128,128)

#losangos
rhombuses16_s = rhombuses_selec.copy()
rhombuses16_n = rhombuses_naoselec.copy()
rhombuses32_s = rhombuses_selec.copy()
rhombuses32_n = rhombuses_naoselec.copy()
rhombuses64_s = rhombuses_selec.copy()
rhombuses64_n = rhombuses_naoselec.copy()
rhombuses128_s = rhombuses_selec.copy()
rhombuses128_n = rhombuses_naoselec.copy()

rhombuses16_s = fun.resizeImages(rhombuses16_s,16,16)
rhombuses16_n = fun.resizeImages(rhombuses16_n,16,16)
rhombuses32_s = fun.resizeImages(rhombuses32_s,32,32)
rhombuses32_n = fun.resizeImages(rhombuses32_n,32,32)
rhombuses64_s = fun.resizeImages(rhombuses64_s,64,64)
rhombuses64_n = fun.resizeImages(rhombuses64_n,64,64)
rhombuses128_s = fun.resizeImages(rhombuses128_s,128,128)
rhombuses128_n = fun.resizeImages(rhombuses128_n,128,128)

#trapézios
trapezia16_s = trapezia_selec.copy()
trapezia16_n = trapezia_naoselec.copy()
trapezia32_s = trapezia_selec.copy()
trapezia32_n = trapezia_naoselec.copy()
trapezia64_s = trapezia_selec.copy()
trapezia64_n = trapezia_naoselec.copy()
trapezia128_s = trapezia_selec.copy()
trapezia128_n = trapezia_naoselec.copy()

trapezia16_s = fun.resizeImages(trapezia16_s,16,16)
trapezia16_n = fun.resizeImages(trapezia16_n,16,16)
trapezia32_s = fun.resizeImages(trapezia32_s,32,32)
trapezia32_n = fun.resizeImages(trapezia32_n,32,32)
trapezia64_s = fun.resizeImages(trapezia64_s,64,64)
trapezia64_n = fun.resizeImages(trapezia64_n,64,64)
trapezia128_s = fun.resizeImages(trapezia128_s,128,128)
trapezia128_n = fun.resizeImages(trapezia128_n,128,128)

#triângulos
triangles16_s = triangles_selec.copy()
triangles16_n = triangles_naoselec.copy()
triangles32_s = triangles_selec.copy()
triangles32_n = triangles_naoselec.copy()
triangles64_s = triangles_selec.copy()
triangles64_n = triangles_naoselec.copy()
triangles128_s = triangles_selec.copy()
triangles128_n = triangles_naoselec.copy()

triangles16_s = fun.resizeImages(triangles16_s,16,16)
triangles16_n = fun.resizeImages(triangles16_n,16,16)
triangles32_s = fun.resizeImages(triangles32_s,32,32)
triangles32_n = fun.resizeImages(triangles32_n,32,32)
triangles64_s = fun.resizeImages(triangles64_s,64,64)
triangles64_n = fun.resizeImages(triangles64_n,64,64)
triangles128_s = fun.resizeImages(triangles128_s,128,128)
triangles128_n = fun.resizeImages(triangles128_n,128,128)

#convertendo para níveis de cinza

squares16_s = fun.grayConversion(squares16_s)
squares16_n = fun.grayConversion(squares16_n)
squares32_s = fun.grayConversion(squares32_s)
squares32_n = fun.grayConversion(squares32_n)
squares64_s = fun.grayConversion(squares64_s)
squares64_n = fun.grayConversion(squares64_n)
squares128_s = fun.grayConversion(squares128_s)
squares128_n = fun.grayConversion(squares128_n)


circles16_s = fun.grayConversion(circles16_s)
circles16_n = fun.grayConversion(circles16_n)
circles32_s = fun.grayConversion(circles32_s)
circles32_n = fun.grayConversion(circles32_n)
circles64_s = fun.grayConversion(circles64_s)
circles64_n = fun.grayConversion(circles64_n)
circles128_s = fun.grayConversion(circles128_s)
circles128_n = fun.grayConversion(circles128_n)

triangles16_s = fun.grayConversion(triangles16_s)
triangles16_n = fun.grayConversion(triangles16_n)
triangles32_s = fun.grayConversion(triangles32_s)
triangles32_n = fun.grayConversion(triangles32_n)
triangles64_s = fun.grayConversion(triangles64_s)
triangles64_n = fun.grayConversion(triangles64_n)
triangles128_s = fun.grayConversion(triangles128_s)
triangles128_n = fun.grayConversion(triangles128_n)


trapezia16_s = fun.grayConversion(trapezia16_s)
trapezia16_n = fun.grayConversion(trapezia16_n)
trapezia32_s = fun.grayConversion(trapezia32_s)
trapezia32_n = fun.grayConversion(trapezia32_n)
trapezia64_s = fun.grayConversion(trapezia64_s)
trapezia64_n = fun.grayConversion(trapezia64_n)
trapezia128_s = fun.grayConversion(trapezia128_s)
trapezia128_n = fun.grayConversion(trapezia128_n)

rhombuses16_s = fun.grayConversion(rhombuses16_s)
rhombuses16_n = fun.grayConversion(rhombuses16_n)
rhombuses32_s = fun.grayConversion(rhombuses32_s)
rhombuses32_n = fun.grayConversion(rhombuses32_n)
rhombuses64_s = fun.grayConversion(rhombuses64_s)
rhombuses64_n = fun.grayConversion(rhombuses64_n)
rhombuses128_s = fun.grayConversion(rhombuses128_s)
rhombuses128_n = fun.grayConversion(rhombuses128_n)

rectangles16_s = fun.grayConversion(rectangles16_s)
rectangles16_n = fun.grayConversion(rectangles16_n)
rectangles32_s = fun.grayConversion(rectangles32_s)
rectangles32_n = fun.grayConversion(rectangles32_n)
rectangles64_s = fun.grayConversion(rectangles64_s)
rectangles64_n = fun.grayConversion(rectangles64_n)
rectangles128_s = fun.grayConversion(rectangles128_s)
rectangles128_n = fun.grayConversion(rectangles128_n)

lines16_s = fun.grayConversion(lines16_s)
lines16_n = fun.grayConversion(lines16_n)
lines32_s = fun.grayConversion(lines32_s)
lines32_n = fun.grayConversion(lines32_n)
lines64_s = fun.grayConversion(lines64_s)
lines64_n = fun.grayConversion(lines64_n)
lines128_s = fun.grayConversion(lines128_s)
lines128_n = fun.grayConversion(lines128_n)

hexagons16_s = fun.grayConversion(hexagons16_s)
hexagons16_n = fun.grayConversion(hexagons16_n)
hexagons32_s = fun.grayConversion(hexagons32_s)
hexagons32_n = fun.grayConversion(hexagons32_n)
hexagons64_s = fun.grayConversion(hexagons64_s)
hexagons64_n = fun.grayConversion(hexagons64_n)
hexagons128_s = fun.grayConversion(hexagons128_s)
hexagons128_n = fun.grayConversion(hexagons128_n)

ellipsis16_s = fun.grayConversion(ellipsis16_s)
ellipsis16_n = fun.grayConversion(ellipsis16_n)
ellipsis32_s = fun.grayConversion(ellipsis32_s)
ellipsis32_n = fun.grayConversion(ellipsis32_n)
ellipsis64_s = fun.grayConversion(ellipsis64_s)
ellipsis64_n = fun.grayConversion(ellipsis64_n)
ellipsis128_s = fun.grayConversion(ellipsis128_s)
ellipsis128_n = fun.grayConversion(ellipsis128_n)

#aplicando o filtro gaussiano

squares16_s = fun.blurConversion(squares16_s,5,0)
squares16_n = fun.blurConversion(squares16_n,5,0)
squares32_s = fun.blurConversion(squares32_s,5,0)
squares32_n = fun.blurConversion(squares32_n,5,0)
squares64_s = fun.blurConversion(squares64_s,5,0)
squares64_n = fun.blurConversion(squares64_n,5,0)
squares128_s = fun.blurConversion(squares128_s,5,0)
squares128_n = fun.blurConversion(squares128_n,5,0)

circles16_s = fun.blurConversion(circles16_s, 5, 0)
circles16_n = fun.blurConversion(circles16_n, 5, 0)
circles32_s = fun.blurConversion(circles32_s,5 ,0)
circles32_n = fun.blurConversion(circles32_n,5 ,0)
circles64_s = fun.blurConversion(circles64_s,5,0)
circles64_n = fun.blurConversion(circles64_n,5,0)
circles128_s = fun.blurConversion(circles128_s,5,0)
circles128_n = fun.blurConversion(circles128_n,5,0)

triangles16_s = fun.blurConversion(triangles16_s,5,0)
triangles16_n = fun.blurConversion(triangles16_n,5,0)
triangles32_s = fun.blurConversion(triangles32_s,5,0)
triangles32_n = fun.blurConversion(triangles32_n,5,0)
triangles64_s = fun.blurConversion(triangles64_s,5,0)
triangles64_n = fun.blurConversion(triangles64_n,5,0)
triangles128_s = fun.blurConversion(triangles128_s,5,0)
triangles128_n = fun.blurConversion(triangles128_n,5,0)

trapezia16_s = fun.blurConversion(trapezia16_s,5,0)
trapezia16_n = fun.blurConversion(trapezia16_n,5,0)
trapezia32_s = fun.blurConversion(trapezia32_s,5,0)
trapezia64_s = fun.blurConversion(trapezia64_s,5,0)
trapezia64_n = fun.blurConversion(trapezia64_n,5,0)
trapezia128_s = fun.blurConversion(trapezia128_s,5,0)
trapezia128_n = fun.blurConversion(trapezia128_n,5,0)

rhombuses16_s = fun.blurConversion(rhombuses16_s,5,0)
rhombuses16_n = fun.blurConversion(rhombuses16_n,5,0)
rhombuses32_s = fun.blurConversion(rhombuses32_s,5,0)
rhombuses32_n = fun.blurConversion(rhombuses32_n,5,0)
rhombuses64_s = fun.blurConversion(rhombuses64_s,5,0)
rhombuses64_n = fun.blurConversion(rhombuses64_n,5,0)
rhombuses128_s = fun.blurConversion(rhombuses128_s,5,0)
rhombuses128_n = fun.blurConversion(rhombuses128_n,5,0)

rectangles16_s = fun.blurConversion(rectangles16_s,5,0)
rectangles16_n = fun.blurConversion(rectangles16_n,5,0)
rectangles32_s = fun.blurConversion(rectangles32_s,5,0)
rectangles32_n = fun.blurConversion(rectangles32_n,5,0)
rectangles64_s = fun.blurConversion(rectangles64_s,5,0)
rectangles64_n = fun.blurConversion(rectangles64_n,5,0)
rectangles128_s = fun.blurConversion(rectangles128_s,5,0)
rectangles128_n = fun.blurConversion(rectangles128_n,5,0)

lines16_s = fun.blurConversion(lines16_s,5,0)
lines16_n = fun.blurConversion(lines16_n,5,0)
lines32_s = fun.blurConversion(lines32_s,5,0)
lines32_n = fun.blurConversion(lines32_n,5,0)
lines64_s = fun.blurConversion(lines64_s,5,0)
lines64_n = fun.blurConversion(lines64_n,5,0)
lines128_s = fun.blurConversion(lines128_s,5,0)
lines128_n = fun.blurConversion(lines128_n,5,0)

hexagons16_s = fun.blurConversion(hexagons16_s,5,0)
hexagons16_n = fun.blurConversion(hexagons16_n,5,0)
hexagons32_s = fun.blurConversion(hexagons32_s,5,0)
hexagons32_n = fun.blurConversion(hexagons32_n,5,0)
hexagons64_s = fun.blurConversion(hexagons64_s,5,0)
hexagons64_n = fun.blurConversion(hexagons64_n,5,0)
hexagons128_s = fun.blurConversion(hexagons128_s,5,0)
hexagons128_n = fun.blurConversion(hexagons128_n,5,0)

ellipsis16_s = fun.blurConversion(ellipsis16_s,5,0)
ellipsis16_n = fun.blurConversion(ellipsis16_n,5,0)
ellipsis32_s = fun.blurConversion(ellipsis32_s,5,0)
ellipsis32_n = fun.blurConversion(ellipsis32_n,5,0)
ellipsis64_s = fun.blurConversion(ellipsis64_s,5,0)
ellipsis64_n = fun.blurConversion(ellipsis64_n,5,0)
ellipsis128_s = fun.blurConversion(ellipsis128_s,5,0)
ellipsis128_n = fun.blurConversion(ellipsis128_n,5,0)


#convertendo para binária
squares16_s = fun.binaryConversion(squares16_s,255,31)
squares16_n = fun.binaryConversion(squares16_n,255,31)
squares32_s = fun.binaryConversion(squares32_s,255,31)
squares32_n = fun.binaryConversion(squares32_n,255,31)
squares64_s = fun.binaryConversion(squares64_s,255,31)
squares64_n = fun.binaryConversion(squares64_n,255,31)
squares128_s = fun.binaryConversion(squares128_s,255,31)
squares128_n = fun.binaryConversion(squares128_n,255,31)

circles16_s = fun.binaryConversion(circles16_s, 255, 31)
circles16_n = fun.binaryConversion(circles16_n, 255, 31)
circles32_s = fun.binaryConversion(circles32_s,255,31)
circles32_n = fun.binaryConversion(circles32_n,255,31)
circles64_s = fun.binaryConversion(circles64_s,255,31)
circles64_n = fun.binaryConversion(circles64_n,255,31)
circles128_s = fun.binaryConversion(circles128_s,255,31)
circles128_n = fun.binaryConversion(circles128_n,255,31)

triangles16_s = fun.binaryConversion(triangles16_s,255,31)
triangles16_n = fun.binaryConversion(triangles16_n,255,31)
triangles32_s = fun.binaryConversion(triangles32_s,255,31)
triangles32_n = fun.binaryConversion(triangles32_n,255,31)
triangles64_s = fun.binaryConversion(triangles64_s,255,31)
triangles64_n = fun.binaryConversion(triangles64_n,255,31)
triangles128_s = fun.binaryConversion(triangles128_s,255,31)
triangles128_n = fun.binaryConversion(triangles128_n,255,31)

trapezia16_s = fun.binaryConversion(trapezia16_s,255,31)
trapezia16_n = fun.binaryConversion(trapezia16_n,255,31)
trapezia32_s = fun.binaryConversion(trapezia32_s,255,31)
trapezia32_n = fun.binaryConversion(trapezia32_n,255,31)
trapezia64_s = fun.binaryConversion(trapezia64_s,255,31)
trapezia64_n = fun.binaryConversion(trapezia64_n,255,31)
trapezia128_s = fun.binaryConversion(trapezia128_s,255,31)
trapezia128_n = fun.binaryConversion(trapezia128_n,255,31)

rhombuses16_s = fun.binaryConversion(rhombuses16_s,255,31)
rhombuses16_n = fun.binaryConversion(rhombuses16_n,255,31)
rhombuses32_s = fun.binaryConversion(rhombuses32_s,255,31)
rhombuses32_n = fun.binaryConversion(rhombuses32_n,255,31)
rhombuses64_s = fun.binaryConversion(rhombuses64_s,255,31)
rhombuses64_n = fun.binaryConversion(rhombuses64_n,255,31)
rhombuses128_s = fun.binaryConversion(rhombuses128_s,255,31)
rhombuses128_n = fun.binaryConversion(rhombuses128_n,255,31)

rectangles16_s = fun.binaryConversion(rectangles16_s,255,31)
rectangles16_n = fun.binaryConversion(rectangles16_n,255,31)
rectangles32_s = fun.binaryConversion(rectangles32_s,255,31)
rectangles32_n = fun.binaryConversion(rectangles32_n,255,31)
rectangles64_s = fun.binaryConversion(rectangles64_s,255,31)
rectangles64_n = fun.binaryConversion(rectangles64_n,255,31)
rectangles128_s = fun.binaryConversion(rectangles128_s,255,31)
rectangles128_n = fun.binaryConversion(rectangles128_n,255,31)

lines16_s = fun.binaryConversion(lines16_s,255,31)
lines16_n = fun.binaryConversion(lines16_n,255,31)
lines32_s = fun.binaryConversion(lines32_s,255,31)
lines32_n = fun.binaryConversion(lines32_n,255,31)
lines64_s = fun.binaryConversion(lines64_s,255,31)
lines64_n = fun.binaryConversion(lines64_n,255,31)
lines128_s = fun.binaryConversion(lines128_s,255,31)
lines128_n = fun.binaryConversion(lines128_n,255,31)

hexagons16_s = fun.binaryConversion(hexagons16_s,255,31)
hexagons16_n = fun.binaryConversion(hexagons16_n,255,31)
hexagons32_s = fun.binaryConversion(hexagons32_s,255,31)
hexagons32_n = fun.binaryConversion(hexagons32_n,255,31)
hexagons64_s = fun.binaryConversion(hexagons64_s,255,31)
hexagons64_n = fun.binaryConversion(hexagons64_n,255,31)
hexagons128_s = fun.binaryConversion(hexagons128_s,255,31)
hexagons128_n = fun.binaryConversion(hexagons128_n,255,31)

ellipsis16_s = fun.binaryConversion(ellipsis16_s,255,31)
ellipsis16_n = fun.binaryConversion(ellipsis16_n,255,31)
ellipsis32_s = fun.binaryConversion(ellipsis32_s,255,31)
ellipsis32_n = fun.binaryConversion(ellipsis32_n,255,31)
ellipsis64_s = fun.binaryConversion(ellipsis64_s,255,31)
ellipsis64_n = fun.binaryConversion(ellipsis64_n,255,31)
ellipsis128_s = fun.binaryConversion(ellipsis128_s,255,31)
ellipsis128_n = fun.binaryConversion(ellipsis128_n,255,31)

#invertendo as cores

squares16_s = fun.invertConversion(squares16_s)
squares16_n = fun.invertConversion(squares16_n)
squares32_s = fun.invertConversion(squares32_s)
squares32_n = fun.invertConversion(squares32_n)
squares64_s = fun.invertConversion(squares64_s)
squares64_n = fun.invertConversion(squares64_n)
squares128_s = fun.invertConversion(squares128_s)
squares128_n = fun.invertConversion(squares128_n)

circles16_s = fun.invertConversion(circles16_s)
circles16_n = fun.invertConversion(circles16_n)
circles32_s = fun.invertConversion(circles32_s)
circles32_n = fun.invertConversion(circles32_n)
circles64_s = fun.invertConversion(circles64_s)
circles64_n = fun.invertConversion(circles64_n)
circles128_s = fun.invertConversion(circles128_s)
circles128_n = fun.invertConversion(circles128_n)

triangles16_s = fun.invertConversion(triangles16_s)
triangles16_n = fun.invertConversion(triangles16_n)
triangles32_s = fun.invertConversion(triangles32_s)
triangles32_n = fun.invertConversion(triangles32_n)
triangles64_s = fun.invertConversion(triangles64_s)
triangles64_n = fun.invertConversion(triangles64_n)
triangles128_s = fun.invertConversion(triangles128_s)
triangles128_n = fun.invertConversion(triangles128_n)

trapezia16_s = fun.invertConversion(trapezia16_s)
trapezia16_n = fun.invertConversion(trapezia16_n)
trapezia32_s = fun.invertConversion(trapezia32_s)
trapezia32_n = fun.invertConversion(trapezia32_n)
trapezia64_s = fun.invertConversion(trapezia64_s)
trapezia64_n = fun.invertConversion(trapezia64_n)
trapezia128_s = fun.invertConversion(trapezia128_s)
trapezia128_n = fun.invertConversion(trapezia128_n)

rhombuses16_s = fun.invertConversion(rhombuses16_s)
rhombuses16_n = fun.invertConversion(rhombuses16_n)
rhombuses32_s = fun.invertConversion(rhombuses32_s)
rhombuses32_n = fun.invertConversion(rhombuses32_n)
rhombuses64_s = fun.invertConversion(rhombuses64_s)
rhombuses64_n = fun.invertConversion(rhombuses64_n)
rhombuses128_s = fun.invertConversion(rhombuses128_s)
rhombuses128_n = fun.invertConversion(rhombuses128_n)

rectangles16_s = fun.invertConversion(rectangles16_s)
rectangles16_n = fun.invertConversion(rectangles16_n)
rectangles32_s = fun.invertConversion(rectangles32_s)
rectangles32_n = fun.invertConversion(rectangles32_n)
rectangles64_s = fun.invertConversion(rectangles64_s)
rectangles64_n = fun.invertConversion(rectangles64_n)
rectangles128_s = fun.invertConversion(rectangles128_s)
rectangles128_n = fun.invertConversion(rectangles128_n)

lines16_s = fun.invertConversion(lines16_s)
lines16_n = fun.invertConversion(lines16_n)
lines32_s = fun.invertConversion(lines32_s)
lines32_n = fun.invertConversion(lines32_n)
lines64_s = fun.invertConversion(lines64_s)
lines64_n = fun.invertConversion(lines64_n)
lines128_s = fun.invertConversion(lines128_s)
lines128_n = fun.invertConversion(lines128_n)

hexagons16_s = fun.invertConversion(hexagons16_s)
hexagons16_n = fun.invertConversion(hexagons16_n)
hexagons32_s = fun.invertConversion(hexagons32_s)
hexagons32_n = fun.invertConversion(hexagons32_n)
hexagons64_s = fun.invertConversion(hexagons64_s)
hexagons64_n = fun.invertConversion(hexagons64_n)
hexagons128_s = fun.invertConversion(hexagons128_s)
hexagons128_n = fun.invertConversion(hexagons128_n)

ellipsis16_s = fun.invertConversion(ellipsis16_s)
ellipsis16_n = fun.invertConversion(ellipsis16_n)
ellipsis32_s = fun.invertConversion(ellipsis32_s)
ellipsis32_n = fun.invertConversion(ellipsis32_n)
ellipsis64_s = fun.invertConversion(ellipsis64_s)
ellipsis64_n = fun.invertConversion(ellipsis64_n)
ellipsis128_s = fun.invertConversion(ellipsis128_s)
ellipsis128_n = fun.invertConversion(ellipsis128_n)
print('terminou pre processing')

# extraindo caracteristicas das imagens

squares128_vector_s = fun.extratorCaracteristicas(squares128_s)
squares128_vector_n = fun.extratorCaracteristicas(squares128_n)
circles128_vector_s = fun.extratorCaracteristicas(circles128_s)
circles128_vector_n = fun.extratorCaracteristicas(circles128_n)
triangles128_vector_s = fun.extratorCaracteristicas(triangles128_s)
triangles128_vector_n = fun.extratorCaracteristicas(triangles128_n)
trapezia128_vector_s = fun.extratorCaracteristicas(trapezia128_s)
trapezia128_vector_n = fun.extratorCaracteristicas(trapezia128_n)
rhombuses128_vector_s = fun.extratorCaracteristicas(rhombuses128_s)
rhombuses128_vector_n = fun.extratorCaracteristicas(rhombuses128_n)
rectangles128_vector_s = fun.extratorCaracteristicas(rectangles128_s)
rectangles128_vector_n = fun.extratorCaracteristicas(rectangles128_n)
lines128_vector_s = fun.extratorCaracteristicas(lines128_s)
lines128_vector_n = fun.extratorCaracteristicas(lines128_n)
hexagons128_vector_s = fun.extratorCaracteristicas(hexagons128_s)
hexagons128_vector_n = fun.extratorCaracteristicas(hexagons128_n)
ellipsis128_vector_s = fun.extratorCaracteristicas(ellipsis128_s)
ellipsis128_vector_n = fun.extratorCaracteristicas(ellipsis128_n)

squares64_vector_s = fun.extratorCaracteristicas(squares64_s)
squares64_vector_n = fun.extratorCaracteristicas(squares64_n)
circles64_vector_s = fun.extratorCaracteristicas(circles64_s)
circles64_vector_n = fun.extratorCaracteristicas(circles64_n)
triangles64_vector_s = fun.extratorCaracteristicas(triangles64_s)
triangles64_vector_n = fun.extratorCaracteristicas(triangles64_n)
trapezia64_vector_s = fun.extratorCaracteristicas(trapezia64_s)
trapezia64_vector_n = fun.extratorCaracteristicas(trapezia64_n)
rhombuses64_vector_s = fun.extratorCaracteristicas(rhombuses64_s)
rhombuses64_vector_n = fun.extratorCaracteristicas(rhombuses64_n)
rectangles64_vector_s = fun.extratorCaracteristicas(rectangles64_s)
rectangles64_vector_n = fun.extratorCaracteristicas(rectangles64_n)
lines64_vector_s = fun.extratorCaracteristicas(lines64_s)
lines64_vector_n = fun.extratorCaracteristicas(lines64_n)
hexagons64_vector_s = fun.extratorCaracteristicas(hexagons64_s)
hexagons64_vector_n = fun.extratorCaracteristicas(hexagons64_n)
ellipsis64_vector_s = fun.extratorCaracteristicas(ellipsis64_s)
ellipsis64_vector_n = fun.extratorCaracteristicas(ellipsis64_n)

squares32_vector_s = fun.extratorCaracteristicas(squares32_s)
squares32_vector_n = fun.extratorCaracteristicas(squares32_n)
circles32_vector_s = fun.extratorCaracteristicas(circles32_s)
circles32_vector_n = fun.extratorCaracteristicas(circles32_n)
triangles32_vector_s = fun.extratorCaracteristicas(triangles32_s)
triangles32_vector_n = fun.extratorCaracteristicas(triangles32_n)
trapezia32_vector_s = fun.extratorCaracteristicas(trapezia32_s)
trapezia32_vector_n = fun.extratorCaracteristicas(trapezia32_n)
rhombuses32_vector_s = fun.extratorCaracteristicas(rhombuses32_s)
rhombuses32_vector_n = fun.extratorCaracteristicas(rhombuses32_n)
rectangles32_vector_s = fun.extratorCaracteristicas(rectangles32_s)
rectangles32_vector_n = fun.extratorCaracteristicas(rectangles32_n)
lines32_vector_s = fun.extratorCaracteristicas(lines32_s)
lines32_vector_n = fun.extratorCaracteristicas(lines32_n)
hexagons32_vector_s = fun.extratorCaracteristicas(hexagons32_s)
hexagons32_vector_n = fun.extratorCaracteristicas(hexagons32_n)
ellipsis32_vector_s = fun.extratorCaracteristicas(ellipsis32_s)
ellipsis32_vector_n = fun.extratorCaracteristicas(ellipsis32_n)


squares16_vector_s = fun.extratorCaracteristicas(squares16_s)
squares16_vector_n = fun.extratorCaracteristicas(squares16_n)
circles16_vector_s = fun.extratorCaracteristicas(circles16_s)
circles16_vector_n = fun.extratorCaracteristicas(circles16_n)
triangles16_vector_s = fun.extratorCaracteristicas(triangles16_s)
triangles16_vector_n = fun.extratorCaracteristicas(triangles16_n)
trapezia16_vector_s = fun.extratorCaracteristicas(trapezia16_s)
trapezia16_vector_n = fun.extratorCaracteristicas(trapezia16_n)
rhombuses16_vector_s = fun.extratorCaracteristicas(rhombuses16_s)
rhombuses16_vector_n = fun.extratorCaracteristicas(rhombuses16_n)
rectangles16_vector_s = fun.extratorCaracteristicas(rectangles16_s)
rectangles16_vector_n = fun.extratorCaracteristicas(rectangles16_n)
lines16_vector_s = fun.extratorCaracteristicas(lines16_s)
lines16_vector_n = fun.extratorCaracteristicas(lines16_n)
hexagons16_vector_s = fun.extratorCaracteristicas(hexagons16_s)
hexagons16_vector_n = fun.extratorCaracteristicas(hexagons16_n)
ellipsis16_vector_s = fun.extratorCaracteristicas(ellipsis16_s)
ellipsis16_vector_n = fun.extratorCaracteristicas(ellipsis16_n)


print('terminou extracao carac')

# transformando os vetores em dataframes


squares128_vector_s = pd.DataFrame(squares128_vector_s)
squares128_vector_n = pd.DataFrame(squares128_vector_n)
circles128_vector_s = pd.DataFrame(circles128_vector_s)
circles128_vector_n = pd.DataFrame(circles128_vector_n)
triangles128_vector_s = pd.DataFrame(triangles128_vector_s)
triangles128_vector_n = pd.DataFrame(triangles128_vector_n)
trapezia128_vector_s = pd.DataFrame(trapezia128_vector_s)
trapezia128_vector_n = pd.DataFrame(trapezia128_vector_n)
rhombuses128_vector_s = pd.DataFrame(rhombuses128_vector_s)
rhombuses128_vector_n = pd.DataFrame(rhombuses128_vector_n)
rectangles128_vector_s = pd.DataFrame(rectangles128_vector_s)
rectangles128_vector_n = pd.DataFrame(rectangles128_vector_n)
lines128_vector_s = pd.DataFrame(lines128_vector_s)
lines128_vector_n = pd.DataFrame(lines128_vector_n)
hexagons128_vector_s = pd.DataFrame(hexagons128_vector_s)
hexagons128_vector_n = pd.DataFrame(hexagons128_vector_n)
ellipsis128_vector_s = pd.DataFrame(ellipsis128_vector_s)
ellipsis128_vector_n = pd.DataFrame(ellipsis128_vector_n)

squares32_vector_s = pd.DataFrame(squares32_vector_s)
squares32_vector_n = pd.DataFrame(squares32_vector_n)
circles32_vector_s = pd.DataFrame(circles32_vector_s)
circles32_vector_n = pd.DataFrame(circles32_vector_n)
triangles32_vector_s = pd.DataFrame(triangles32_vector_s)
triangles32_vector_n = pd.DataFrame(triangles32_vector_n)
trapezia32_vector_s = pd.DataFrame(trapezia32_vector_s)
trapezia32_vector_n = pd.DataFrame(trapezia32_vector_n)
rhombuses32_vector_s = pd.DataFrame(rhombuses32_vector_s)
rhombuses32_vector_n = pd.DataFrame(rhombuses32_vector_n)
rectangles32_vector_s = pd.DataFrame(rectangles32_vector_s)
rectangles32_vector_n = pd.DataFrame(rectangles32_vector_n)
hexagons32_vector_s = pd.DataFrame(hexagons32_vector_s)
hexagons32_vector_n = pd.DataFrame(hexagons32_vector_n)
ellipsis32_vector_s = pd.DataFrame(ellipsis32_vector_s)
ellipsis32_vector_n = pd.DataFrame(ellipsis32_vector_n)
lines32_vector_s = pd.DataFrame(lines32_vector_s)
lines32_vector_n = pd.DataFrame(lines32_vector_n)

squares64_vector_s = pd.DataFrame(squares64_vector_s)
squares64_vector_n = pd.DataFrame(squares64_vector_n)
circles64_vector_s = pd.DataFrame(circles64_vector_s)
circles64_vector_n = pd.DataFrame(circles64_vector_n)
triangles64_vector_s = pd.DataFrame(triangles64_vector_s)
triangles64_vector_n = pd.DataFrame(triangles64_vector_n)
trapezia64_vector_s = pd.DataFrame(trapezia64_vector_s)
trapezia64_vector_n = pd.DataFrame(trapezia64_vector_n)
rhombuses64_vector_s = pd.DataFrame(rhombuses64_vector_s)
rhombuses64_vector_n = pd.DataFrame(rhombuses64_vector_n)
rectangles64_vector_s = pd.DataFrame(rectangles64_vector_s)
rectangles64_vector_n = pd.DataFrame(rectangles64_vector_n)
lines64_vector_s = pd.DataFrame(lines64_vector_s)
lines64_vector_n = pd.DataFrame(lines64_vector_n)
hexagons64_vector_s = pd.DataFrame(hexagons64_vector_s)
hexagons64_vector_n = pd.DataFrame(hexagons64_vector_n)
ellipsis64_vector_s = pd.DataFrame(ellipsis64_vector_s)
ellipsis64_vector_n = pd.DataFrame(ellipsis64_vector_n)


circles16_vector_s = pd.DataFrame(circles16_vector_s)
circles16_vector_n = pd.DataFrame(circles16_vector_n)
squares16_vector_s = pd.DataFrame(squares16_vector_s)
squares16_vector_n = pd.DataFrame(squares16_vector_n)
triangles16_vector_s = pd.DataFrame(triangles16_vector_s)
triangles16_vector_n = pd.DataFrame(triangles16_vector_n)
trapezia16_vector_s = pd.DataFrame(trapezia16_vector_s)
trapezia16_vector_n = pd.DataFrame(trapezia16_vector_n)
rhombuses16_vector_s = pd.DataFrame(rhombuses16_vector_s)
rhombuses16_vector_n = pd.DataFrame(rhombuses16_vector_n)
rectangles16_vector_s = pd.DataFrame(rectangles16_vector_s)
rectangles16_vector_n = pd.DataFrame(rectangles16_vector_n)
lines16_vector_s = pd.DataFrame(lines16_vector_s)
lines16_vector_n = pd.DataFrame(lines16_vector_n)
hexagons16_vector_s = pd.DataFrame(hexagons16_vector_s)
hexagons16_vector_n = pd.DataFrame(hexagons16_vector_n)
ellipsis16_vector_s = pd.DataFrame(ellipsis16_vector_s)
ellipsis16_vector_n = pd.DataFrame(ellipsis16_vector_n)


print('terminou transformar em dataframe')

#incluindo a classe nos dataframes

squares128_vector_s['Classe'] = 'square'
squares128_vector_n['Classe'] = 'square'
circles128_vector_s['Classe'] = 'circle'
circles128_vector_n['Classe'] = 'circle'
triangles128_vector_s['Classe'] = 'triangle'
triangles128_vector_n['Classe'] = 'triangle'
trapezia128_vector_s['Classe'] = 'trapezia'
trapezia128_vector_n['Classe'] = 'trapezia'
rhombuses128_vector_s['Classe'] = 'rhombuse'
rhombuses128_vector_n['Classe'] = 'rhombuse'
rectangles128_vector_s['Classe'] = 'rectangle'
rectangles128_vector_n['Classe'] = 'rectangle'
lines128_vector_s['Classe'] = 'line'
lines128_vector_n['Classe'] = 'line'
hexagons128_vector_s['Classe'] = 'hexagon'
hexagons128_vector_n['Classe'] = 'hexagon'
ellipsis128_vector_s['Classe'] = 'ellipse'
ellipsis128_vector_n['Classe'] = 'ellipse'

squares32_vector_s['Classe'] = 'square'
squares32_vector_n['Classe'] = 'square'
circles32_vector_s['Classe'] = 'circle'
circles32_vector_n['Classe'] = 'circle'
triangles32_vector_s['Classe'] = 'triangle'
triangles32_vector_n['Classe'] = 'triangle'
trapezia32_vector_s['Classe'] = 'trapezia'
trapezia32_vector_n['Classe'] = 'trapezia'
rhombuses32_vector_s['Classe'] = 'rhombuse'
rhombuses32_vector_n['Classe'] = 'rhombuse'
rectangles32_vector_s['Classe'] = 'rectangle'
rectangles32_vector_n['Classe'] = 'rectangle'
lines32_vector_s['Classe'] = 'line'
lines32_vector_n['Classe'] = 'line'
hexagons32_vector_s['Classe'] = 'hexagon'
hexagons32_vector_n['Classe'] = 'hexagon'
ellipsis32_vector_s['Classe'] = 'ellipse'
ellipsis32_vector_n['Classe'] = 'ellipse'

squares64_vector_s['Classe'] = 'square'
squares64_vector_n['Classe'] = 'square'
circles64_vector_s['Classe'] = 'circle'
circles64_vector_n['Classe'] = 'circle'
triangles64_vector_s['Classe'] = 'triangle'
triangles64_vector_n['Classe'] = 'triangle'
trapezia64_vector_s['Classe'] = 'trapezia'
trapezia64_vector_n['Classe'] = 'trapezia'
rhombuses64_vector_s['Classe'] = 'rhombuse'
rhombuses64_vector_n['Classe'] = 'rhombuse'
rectangles64_vector_s['Classe'] = 'rectangle'
rectangles64_vector_n['Classe'] = 'rectangle'
lines64_vector_s['Classe'] = 'line'
lines64_vector_n['Classe'] = 'line'
hexagons64_vector_s['Classe'] = 'hexagon'
hexagons64_vector_n['Classe'] = 'hexagon'
ellipsis64_vector_s['Classe'] = 'ellipse'
ellipsis64_vector_n['Classe'] = 'ellipse'


squares16_vector_s['Classe'] = 'square'
squares16_vector_n['Classe'] = 'square'
circles16_vector_s['Classe'] = 'circle'
circles16_vector_n['Classe'] = 'circle'
triangles16_vector_s['Classe'] = 'triangle'
triangles16_vector_n['Classe'] = 'triangle'
trapezia16_vector_s['Classe'] = 'trapezia'
trapezia16_vector_n['Classe'] = 'trapezia'
rhombuses16_vector_s['Classe'] = 'rhombuse'
rhombuses16_vector_n['Classe'] = 'rhombuse'
rectangles16_vector_s['Classe'] = 'rectangle'
rectangles16_vector_n['Classe'] = 'rectangle'
lines16_vector_s['Classe'] = 'line'
lines16_vector_n['Classe'] = 'line'
hexagons16_vector_s['Classe'] = 'hexagon'
hexagons16_vector_n['Classe'] = 'hexagon'
ellipsis16_vector_s['Classe'] = 'ellipse'
ellipsis16_vector_n['Classe'] = 'ellipse'



dfs64_s = [squares64_vector_s,circles64_vector_s,triangles64_vector_s,trapezia64_vector_s,rhombuses64_vector_s,
         rectangles64_vector_s,lines64_vector_s,hexagons64_vector_s,ellipsis64_vector_s]

dfs64_n = [squares64_vector_n,circles64_vector_n,triangles64_vector_n,trapezia64_vector_n,rhombuses64_vector_n,
         rectangles64_vector_n,lines64_vector_n,hexagons64_vector_n,ellipsis64_vector_n]

dfs128_s = [squares128_vector_s,circles128_vector_s,triangles128_vector_s,trapezia128_vector_s,rhombuses128_vector_s,
          rectangles128_vector_s,lines128_vector_s,hexagons128_vector_s,ellipsis128_vector_s]

dfs128_n = [squares128_vector_n,circles128_vector_n,triangles128_vector_n,trapezia128_vector_n,rhombuses128_vector_n,
          rectangles128_vector_n,lines128_vector_n,hexagons128_vector_n,ellipsis128_vector_n]

dfs32_s = [squares32_vector_s,circles32_vector_s,triangles32_vector_s,trapezia32_vector_s,rhombuses32_vector_s,
         rectangles32_vector_s,lines32_vector_s,hexagons32_vector_s,ellipsis32_vector_s]

dfs32_n = [squares32_vector_n,circles32_vector_n,triangles32_vector_n,trapezia32_vector_n,rhombuses32_vector_n,
         rectangles32_vector_n,lines32_vector_n,hexagons32_vector_n,ellipsis32_vector_n]

dfs16_s = [squares16_vector_s,circles16_vector_s,triangles16_vector_s,trapezia16_vector_s,rhombuses16_vector_s,
       rectangles16_vector_s,lines16_vector_s,hexagons16_vector_s,ellipsis16_vector_s]
dfs16_n = [squares16_vector_n,circles16_vector_n,triangles16_vector_n,trapezia16_vector_n,rhombuses16_vector_n,
       rectangles16_vector_n,lines16_vector_n,hexagons16_vector_n,ellipsis16_vector_n]



# USANDO AS IMAGENS 128x128

dataFrame128_s = pd.concat(dfs128_s, ignore_index=True)
dataFrame128_2_s = dataFrame128_s.copy()
del dataFrame128_s['Classe']
# dataFrame128_s.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Images_128_s_NOCLASS.csv')
dataFrame128_s = fun.normalizar(dataFrame128_s)
dataFrame128_s.fillna(0)
dataFrame128_s['Classe'] = dataFrame128_2_s['Classe']
dataFrame128_s.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Images_128_s.csv')

dataFrame128_n = pd.concat(dfs128_n, ignore_index=True)
dataFrame128_n.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Images_128_n.csv')
dataFrame128_2_n = dataFrame128_n.copy()
del dataFrame128_n['Classe']
dataFrame128_n = fun.normalizar(dataFrame128_n)
dataFrame128_n.fillna(0)
dataFrame128_n['Classe'] = dataFrame128_2_n['Classe']

dataFrame64_s = pd.concat(dfs64_s, ignore_index=True)
dataFrame64_s = dataFrame64_s.fillna(0)
dataFrame64_2_s = dataFrame64_s.copy()
del dataFrame64_s['Classe']
# dataFrame64_s.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Images_64_sNOCLASS.csv')
dataFrame64_s = fun.normalizar(dataFrame64_s)
dataFrame64_s['Classe'] = dataFrame64_2_s['Classe']
dataFrame64_s.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Images_64_s.csv')

dataFrame64_n = pd.concat(dfs64_n, ignore_index=True)
dataFrame64_n.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Images_64_n.csv')
dataFrame64_n = dataFrame64_n.fillna(0)
dataFrame64_2_n = dataFrame64_n.copy()
del dataFrame64_n['Classe']
dataFrame64_n = fun.normalizar(dataFrame64_n)
dataFrame64_n['Classe'] = dataFrame64_2_n['Classe']

dataFrame32_s = pd.concat(dfs32_s, ignore_index=True)
dataFrame32_s = dataFrame32_s.fillna(0)
dataFrame32_2_s = dataFrame32_s.copy()
del dataFrame32_s['Classe']
# dataFrame32_s.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Images_32_sNOCLASS.csv')
dataFrame32_s = fun.normalizar(dataFrame32_s)
dataFrame32_s['Classe'] = dataFrame32_2_s['Classe']
dataFrame32_s.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Images_32_s.csv')

dataFrame32_n = pd.concat(dfs32_n, ignore_index=True)
dataFrame32_n.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Images_32_n.csv')
dataFrame32_n = dataFrame32_n.fillna(0)
dataFrame32_2_n = dataFrame32_n.copy()
del dataFrame32_n['Classe']
dataFrame32_n = fun.normalizar(dataFrame32_n)
dataFrame32_n['Classe'] = dataFrame32_2_n['Classe']

dataFrame16_s = pd.concat(dfs16_s, ignore_index=True)
dataFrame16_s = dataFrame16_s.fillna(0)
dataFrame16_2_s = dataFrame16_s.copy()
del dataFrame16_s['Classe']
# dataFrame16_s.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Images_16_sNOCLASS.csv')
dataFrame16_s = fun.normalizar(dataFrame16_s)
dataFrame16_s['Classe'] = dataFrame16_2_s['Classe']
dataFrame16_s.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Images_16_s.csv')

dataFrame16_n = pd.concat(dfs16_n, ignore_index=True)
dataFrame16_n.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Images_16_n.csv')
dataFrame16_n = dataFrame16_n.fillna(0)
dataFrame16_2_n = dataFrame16_n.copy()
del dataFrame16_n['Classe']
dataFrame16_n = fun.normalizar(dataFrame16_n)
dataFrame16_n['Classe'] = dataFrame16_2_n['Classe']

# Criando o objeto do tipo k-folds com 10 folds
# kfold = KFold(10, True, 1)


# Criando o k-fold com 5 folds para execução do algoritmo genético
kfold = KFold(5, True, 1)

#Inicializando o Classificador do algoritmo genético

# Random Forest Classifier
RandomForest = RandomForestClassifier()
RandomForest_acerto = []
RandomForest_accmedia = []

#criando a população com 20 cromossomos de tamanho 38
cromossomos = fun.create_population(20, 38)


for cromo in cromossomos:
    positions = fun.positions_chromossome(cromo)
    df_classe = fun.decode_chromossome(cromo)
    df_classe_2 = df_classe.copy()
    del df_classe['Classe']
    imagens = np.array(df_classe)
    caracteristicas = fun.carac_imagens(positions, imagens)
    for x in range(0, 5):

        tempo1 = time.time()
        cols = list(df_classe_2.columns)
        cols.remove('Classe')
        df_images_noclass = df_classe_2[cols]
        df_images_class = df_classe_2['Classe']
        c = kfold.split(df_classe_2)

        for train_index, test_index in c:
            noclass_train, noclass_test = df_images_noclass.iloc[train_index], df_images_noclass.iloc[test_index]
            class_train, class_test = df_images_class.iloc[train_index], df_images_class.iloc[test_index]

            RandomForest_inicio = time.time()
            RandomForest.fit(noclass_train, class_train)
            RandomForest_acerto.append(RandomForest.score(noclass_test, class_test))

        df_classe_2 = df_classe_2.sample(frac=1)
        print("Terminou a ", x)
        tempo2 = time.time()
        print("Tempo da rodada ", x, (tempo2 - tempo1) / 60)

    RandomForest_accmedia.append(np.mean(RandomForest_acerto))

    tempofinal = time.time()

print('acuracia media ', RandomForest_accmedia)
print('acuracia')
print(RandomForest_acerto)
print('cromossomos')
print(cromossomos)

# passando a função que retorna os dois melhores individuos, as suas acuracias e o array de cromossomos
#atualizado
melhores_ind, best_acuracia, cromossomos = fun.get_best_cromossomos(RandomForest_accmedia,cromossomos)

#realizando o torneio para selecionar 10 pais
pais_torneio = []
for i in range(0,10):
    aux, cromossomos = fun.tournament_selection(RandomForest_accmedia,cromossomos)
    pais_torneio.append(aux)


#escolhendo os pais aleatoriamente
pais_pares = fun.generate_parents(pais_torneio)


#gerando filhos com operador crossover
offspring = []
for x in range(0, 5):
    filhos = []
    filhos.append(fun.crossover(0.9, pais_pares[x]))

    for f in filhos:

        f = fun.mutation(0.05, f)
        offspring.append(f)


#garantindo o elitismo
for ind in melhores_ind:
    offspring.append(ind)


print('offspring')
print(offspring)
print('melhores individios')
print(melhores_ind)

# melhores, acuracias_best, cromossomos = fun.get_best_cromossomos(RandomForest_acerto, cromossomos)
# print('melhores cromossomos', melhores)
# print('melhores acuracias', acuracias_best)
# filhos,pais = fun.genetic_algorithm(melhores,cromossomos,RandomForest_acerto)
# print('filhos')
# print(filhos)
# print('pais')
# print(pais)

# Instanciando os algoritmos e seus vetores de tempo e acurácia
'''
#instanciando DataFrame com dados de saida
DadosSaida = []
DadosSaida = pd.DataFrame(DadosSaida)


# Random Forest Classifier
RandomForest = RandomForestClassifier()
RandomForest_acerto = []
RandomForest_tempo = []
RandomForest_precision = []
RandomForest_recall = []
RandomForest_fscore = []

# SVM com função de kernel linear
SVMachine_L = SVC(kernel='linear')
SVMachine_L_acerto = []
SVMachine_L_tempo = []
SVMachine_L_precision = []
SVMachine_L_recall = []
SVMachine_L_fscore = []

#SVM com função de kernel RBF
SVMachine_RBF = SVC(kernel='rbf', gamma='scale')
SVMachine_RBF_acerto = []
SVMachine_RBF_tempo = []
SVMachine_RBF_precision = []
SVMachine_RBF_recall = []
SVMachine_RBF_fscore = []

# KNN com k = 3, 5, 7
KNN_3 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
KNN_3_acerto = []
KNN_3_tempo = []
KNN_3_precision = []
KNN_3_recall = []
KNN_3_fscore = []

KNN_5 = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
KNN_5_acerto = []
KNN_5_tempo = []
KNN_5_precision = []
KNN_5_recall = []
KNN_5_fscore = []

KNN_7 =  KNeighborsClassifier(n_neighbors=7, metric='euclidean')
KNN_7_acerto = []
KNN_7_tempo = []
KNN_7_precision = []
KNN_7_recall = []
KNN_7_fscore = []

# KNN Ponderado com k = 3, 5, 7
KNNP_3 = KNeighborsClassifier(n_neighbors=3, weights='distance',metric='euclidean')
KNNP_3_acerto = []
KNNP_3_tempo = []
KNNP_3_precision = []
KNNP_3_recall = []
KNNP_3_fscore = []

KNNP_5 = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
KNNP_5_acerto = []
KNNP_5_tempo = []
KNNP_5_precision = []
KNNP_5_recall = []
KNNP_5_fscore = []

KNNP_7 = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')
KNNP_7_acerto = []
KNNP_7_tempo = []
KNNP_7_precision = []
KNNP_7_recall = []
KNNP_7_fscore = []

# Naïve Bayes
NaiveBayes = GaussianNB()
NaiveBayes_acerto = []
NaiveBayes_tempo = []
NaiveBayes_precision = []
NaiveBayes_recall = []
NaiveBayes_fscore = []

# Árvore de decisão
DecisionTree = DecisionTreeClassifier()
DecisionTree_acerto = []
DecisionTree_tempo = []
DecisionTree_precision = []
DecisionTree_recall = []
DecisionTree_fscore = []

# MultiLayer Perceptron
MLP = MLPClassifier()
MLP_acerto = []
MLP_tempo = []
MLPP_precision = []
MLP_recall = []
MLP_fscore = []

# Regressão Logística
RegrLogistica = LogisticRegression(solver='lbfgs')
RegrLogistica_acerto = []
RegrLogistica_tempo = []
RegreLogistica_precision = []
RegrLogistica_recall = []
RegrLogistica_fscore = []

# ____________________ USANDO IMAGENS 128x128

print('comecou o K fold')

tempoinicial = time.time()

for x in range(0, 10):

    tempo1 = time.time()
    cols = list(dataFrame128.columns)
    cols.remove('Classe')
    df_images_noclass = dataFrame128[cols]
    df_images_class = dataFrame128['Classe']
    c = kfold.split(dataFrame128)

    for train_index, test_index in c:

        noclass_train, noclass_test = df_images_noclass.iloc[train_index], df_images_noclass.iloc[test_index]
        class_train, class_test = df_images_class.iloc[train_index], df_images_class.iloc[test_index]

        KNN3_inicio = time.time()
        KNN_3.fit(noclass_train, class_train)
        KNN_3_acerto.append(KNN_3.score(noclass_test, class_test))
        KNN_3_recall. append(recall_score(class_test, KNN_3.predict(noclass_test),average='weighted'))
        KNN_3_precision.append(precision_score(class_test,KNN_3.predict(noclass_test),average='weighted'))
        KNN_3_fscore.append(f1_score(class_test,KNN_3.predict(noclass_test),average='weighted'))
        KNN3_fim = time.time()
        KNN_3_tempo.append(KNN3_fim - KNN3_inicio)

        KNN5_inicio = time.time()
        KNN_5.fit(noclass_train, class_train)
        KNN_5_acerto.append(KNN_5.score(noclass_test, class_test))
        KNN_5_recall.append(recall_score(class_test,KNN_5.predict(noclass_test),average='weighted'))
        KNN_5_precision.append(precision_score(class_test,KNN_5.predict(noclass_test),average='weighted'))
        KNN_5_fscore.append(f1_score(class_test,KNN_5.predict(noclass_test),average='weighted'))
        KNN5_fim = time.time()
        KNN_5_tempo.append(KNN5_fim - KNN5_inicio)



        KNN7_inicio = time.time()
        KNN_7.fit(noclass_train, class_train)
        KNN_7_acerto.append(KNN_3.score(noclass_test, class_test))
        KNN_7_precision.append(precision_score(class_test,KNN_7.predict(noclass_test),average='weighted'))
        KNN_7_recall.append(recall_score(class_test,KNN_7.predict(noclass_test),average='weighted'))
        KNN_7_fscore.append(f1_score(class_test,KNN_7.predict(noclass_test),average='weighted'))
        KNN7_fim = time.time()
        KNN_7_tempo.append(KNN3_fim - KNN3_inicio)

        KNNP3_inicio = time.time()
        KNNP_3.fit(noclass_train, class_train)
        KNNP_3_acerto.append(KNNP_3.score(noclass_test, class_test))
        KNNP_3_precision.append(precision_score(class_test,KNNP_3.predict(noclass_test),average='weighted'))
        KNNP_3_recall.append(recall_score(class_test,KNNP_3.predict(noclass_test),average='weighted'))
        KNNP_3_fscore.append(f1_score(class_test,KNNP_3.predict(noclass_test),average='weighted'))
        KNNP3_fim = time.time()
        KNNP_3_tempo.append(KNNP3_fim - KNNP3_inicio)

        KNNP5_inicio = time.time()
        KNNP_5.fit(noclass_train, class_train)
        KNNP_5_acerto.append(KNNP_5.score(noclass_test, class_test))
        KNNP_5_precision.append(precision_score(class_test,KNNP_5.predict(noclass_test),average='weighted'))
        KNNP_5_recall.append(recall_score(class_test,KNNP_5.predict(noclass_test),average='weighted'))
        KNNP_5_fscore.append(f1_score(class_test,KNNP_5.predict(noclass_test),average='weighted'))
        KNNP5_fim = time.time()
        KNNP_5_tempo.append(KNNP5_fim - KNNP5_inicio)

        KNNP7_inicio = time.time()
        KNNP_7.fit(noclass_train, class_train)
        KNNP_7_acerto.append(KNNP_7.score(noclass_test, class_test))
        KNNP_7_precision.append(precision_score(class_test,KNNP_7.predict(noclass_test),average='weighted'))
        KNNP_7_recall.append(recall_score(class_test,KNNP_7.predict(noclass_test),average='weighted'))
        KNNP_7_fscore.append(f1_score(class_test,KNNP_7.predict(noclass_test),average='weighted'))
        KNNP7_fim = time.time()
        KNNP_7_tempo.append(KNNP7_fim - KNNP7_inicio)

        NaiveBayes_inicio = time.time()
        NaiveBayes.fit(noclass_train, class_train)
        NaiveBayes_acerto.append(NaiveBayes.score(noclass_test, class_test))
        NaiveBayes_precision.append(precision_score(class_test,NaiveBayes.predict(noclass_test),average='weighted'))
        NaiveBayes_recall.append(recall_score(class_test,NaiveBayes.predict(noclass_test),average='weighted'))
        NaiveBayes_fscore.append(f1_score(class_test,NaiveBayes.predict(noclass_test),average='weighted'))
        NaiveBayes_fim = time.time()
        NaiveBayes_tempo.append(NaiveBayes_fim - NaiveBayes_inicio)

        DecisionTree_inicio = time.time()
        DecisionTree.fit(noclass_train, class_train)
        DecisionTree_acerto.append(DecisionTree.score(noclass_test, class_test))
        DecisionTree_precision.append(precision_score(class_test,DecisionTree.predict(noclass_test),average='weighted'))
        DecisionTree_recall.append(recall_score(class_test,DecisionTree.predict(noclass_test),average='weighted'))
        DecisionTree_fscore.append(f1_score(class_test,DecisionTree.predict(noclass_test),average='weighted'))
        DecisionTree_fim = time.time()
        DecisionTree_tempo.append(DecisionTree_fim - DecisionTree_inicio)

        SVMachine_L_inicio = time.time()
        SVMachine_L.fit(noclass_train, class_train)
        SVMachine_L_acerto.append(SVMachine_L.score(noclass_test, class_test))
        SVMachine_L_precision.append(precision_score(class_test,SVMachine_L.predict(noclass_test),average='weighted'))
        SVMachine_L_recall.append(recall_score(class_test,SVMachine_L.predict(noclass_test),average='weighted'))
        SVMachine_L_fscore.append(f1_score(class_test,SVMachine_L.predict(noclass_test),average='weighted'))
        SVMachine_L_fim = time.time()
        SVMachine_L_tempo.append(SVMachine_L_fim - SVMachine_L_inicio)

        SVMachine_RBF_inicio = time.time()
        SVMachine_RBF.fit(noclass_train, class_train)
        SVMachine_RBF_acerto.append(SVMachine_RBF.score(noclass_test, class_test))
        SVMachine_RBF_recall.append(recall_score(class_test,SVMachine_RBF.predict(noclass_test),average='weighted'))
        SVMachine_RBF_precision.append(precision_score(class_test,SVMachine_RBF.predict(noclass_test),average='weighted'))
        SVMachine_RBF_fscore.append(f1_score(class_test,SVMachine_RBF.predict(noclass_test),average='weighted'))
        SVMachine_RBF_fim = time.time()
        SVMachine_RBF_tempo.append(SVMachine_RBF_fim - SVMachine_RBF_inicio)

        RegrLogistica_inicio = time.time()
        RegrLogistica.fit(noclass_train, class_train)
        RegrLogistica_acerto.append(RegrLogistica.score(noclass_test, class_test))
        RegreLogistica_precision.append(precision_score(class_test, (RegrLogistica.predict(noclass_test)),average='weighted'))
        RegrLogistica_recall.append(recall_score(class_test,RegrLogistica.predict(noclass_test),average='weighted'))
        RegrLogistica_fscore.append(f1_score(class_test,RegrLogistica.predict(noclass_test),average='weighted'))
        RegrLogistica_fim = time.time()
        RegrLogistica_tempo.append(RegrLogistica_fim - RegrLogistica_inicio)

        MLP_inicio = time.time()
        MLP.fit(noclass_train, class_train)
        MLP_acerto.append(MLP.score(noclass_test, class_test))
        MLPP_precision.append(precision_score(class_test,MLP.predict(noclass_test),average='weighted'))
        MLP_recall.append(recall_score(class_test,MLP.predict(noclass_test),average='weighted'))
        MLP_fscore.append(f1_score(class_test,MLP.predict(noclass_test),average='weighted'))
        MLP_fim = time.time()
        MLP_tempo.append(MLP_fim - MLP_inicio)

        RandomForest_inicio = time.time()
        RandomForest.fit(noclass_train, class_train)
        RandomForest_acerto.append(RandomForest.score(noclass_test, class_test))
        RandomForest_recall.append(recall_score(class_test,RandomForest.predict(noclass_test),average='weighted'))
        RandomForest_precision.append(precision_score(class_test,RandomForest.predict(noclass_test),average='weighted'))
        RandomForest_fscore.append(f1_score(class_test,RandomForest.predict(noclass_test),average='weighted'))
        RandomForest_fim = time.time()
        RandomForest_tempo.append(RandomForest_fim - RandomForest_inicio)


    dataFrame128 = dataFrame128.sample(frac=1)
    print("Terminou a ", x)
    tempo2 = time.time()
    print("Tempo da rodada ", x, (tempo2 - tempo1) / 60)

tempofinal = time.time()

fun.tendencia_central('KNN k = 3', KNN_3_acerto, KNN_3_tempo)
fun.tendencia_central('KNN k = 5', KNN_5_acerto, KNN_5_tempo)
fun.tendencia_central('KNN k = 7', KNN_7_acerto, KNN_7_tempo)
fun.tendencia_central('KNN Ponderado k = 3', KNNP_3_acerto, KNNP_3_tempo)
fun.tendencia_central('KNN Ponderado k = 5', KNNP_5_acerto, KNNP_5_tempo)
fun.tendencia_central('KNN Ponderado k = 7', KNNP_7_acerto, KNNP_7_tempo)
fun.tendencia_central('Naïve Bayes', NaiveBayes_acerto, NaiveBayes_tempo)
fun.tendencia_central('Árvore de decisão', DecisionTree_acerto, DecisionTree_tempo)
fun.tendencia_central('MLP', MLP_acerto, MLP_tempo)
fun.tendencia_central('Regressão Logística', RegrLogistica_acerto, RegrLogistica_tempo)
fun.tendencia_central('SVM linear', SVMachine_L_acerto, SVMachine_L_tempo)
fun.tendencia_central('SVM RBF', SVMachine_RBF_acerto, SVMachine_RBF_tempo)
fun.tendencia_central('Random Forest', RandomForest_acerto,RandomForest_tempo)

Acuracia128 = [KNN_3_acerto,KNN_5_acerto,KNN_7_acerto,KNNP_3_acerto,KNNP_5_acerto,KNNP_7_acerto,
               NaiveBayes_acerto,DecisionTree_acerto,MLP_acerto,RegrLogistica_acerto,SVMachine_L_acerto,
               SVMachine_RBF_acerto,RandomForest_acerto]
Precision128 = [KNN_3_precision,KNN_5_precision,KNN_7_precision,KNNP_3_precision,KNNP_5_precision,KNNP_7_precision,
         NaiveBayes_precision,DecisionTree_precision,MLPP_precision,RegreLogistica_precision,SVMachine_L_precision,
         SVMachine_RBF_precision,RandomForest_precision]
Recall128 = [KNN_3_recall,KNN_5_recall,KNN_7_recall,KNNP_3_recall,KNNP_5_recall,KNNP_7_recall,NaiveBayes_recall,
          DecisionTree_recall,MLP_recall,RegrLogistica_recall,SVMachine_L_recall,SVMachine_RBF_recall,RandomForest_recall]

Fscore128 = [KNN_3_fscore,KNN_5_fscore,KNN_7_fscore,KNNP_3_fscore,KNN_5_fscore,KNN_7_fscore,NaiveBayes_fscore,DecisionTree_fscore,
          MLP_fscore,RegrLogistica_fscore,SVMachine_L_fscore, SVMachine_RBF_fscore,RandomForest_fscore]

Acuracia128 = pd.DataFrame(Acuracia128)
Precision128 = pd.DataFrame(Precision128)
Recall128 = pd.DataFrame(Recall128)
Fscore128 = pd.DataFrame(Fscore128)

Acuracia128.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Acuracia128.csv')
Precision128.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Precision128.csv')
Recall128.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Recall128.csv')
Fscore128.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Fscore128.csv')

mediasacuracias = {


    "KNN Ponderado k = 3": np.mean(KNNP_3_acerto),
    "KNN Ponderado k = 5": np.mean(KNNP_5_acerto),
    "KNN Ponderado k = 7": np.mean(KNNP_7_acerto),
    "Naive Bayes": np.mean(NaiveBayes_acerto),
    "KNN k = 3": np.mean(KNN_3_acerto),
    "KNN k = 5": np.mean(KNN_5_acerto),
    "KNN k = 7": np.mean(KNN_7_acerto),
    "Decision Tree": np.mean(DecisionTree_acerto),
    "SVM Linear": np.mean(SVMachine_L_acerto),
    "SVM RBF": np.mean(SVMachine_RBF_acerto),
    "Regressao Logistica": np.mean(RegrLogistica_acerto),
    "MLP": np.mean(MLP_acerto),
    "Random Forest": np.mean(RandomForest_acerto)
}



mediasacuracias = sorted(mediasacuracias.items(),
                         key=lambda x: x[1])
print(mediasacuracias)
print("Tempo total: ", (tempofinal - tempoinicial) / 60)


## USANDO A BASE COM IMAGENS DE 64x64



print('comecou o K fold')

tempoinicial = time.time()

for x in range(0, 10):

    tempo1 = time.time()
    cols = list(dataFrame64.columns)
    cols.remove('Classe')
    df_images_noclass = dataFrame64[cols]
    df_images_class = dataFrame64['Classe']
    c = kfold.split(dataFrame64)

    for train_index, test_index in c:

        noclass_train, noclass_test = df_images_noclass.iloc[train_index], df_images_noclass.iloc[test_index]
        class_train, class_test = df_images_class.iloc[train_index], df_images_class.iloc[test_index]

        KNN3_inicio = time.time()
        KNN_3.fit(noclass_train, class_train)
        KNN_3_acerto.append(KNN_3.score(noclass_test, class_test))
        KNN_3_recall. append(recall_score(class_test, KNN_3.predict(noclass_test),average='weighted'))
        KNN_3_precision.append(precision_score(class_test,KNN_3.predict(noclass_test),average='weighted'))
        KNN_3_fscore.append(f1_score(class_test,KNN_3.predict(noclass_test),average='weighted'))
        KNN3_fim = time.time()
        KNN_3_tempo.append(KNN3_fim - KNN3_inicio)

        KNN5_inicio = time.time()
        KNN_5.fit(noclass_train, class_train)
        KNN_5_acerto.append(KNN_5.score(noclass_test, class_test))
        KNN_5_recall.append(recall_score(class_test,KNN_5.predict(noclass_test),average='weighted'))
        KNN_5_precision.append(precision_score(class_test,KNN_5.predict(noclass_test),average='weighted'))
        KNN_5_fscore.append(f1_score(class_test,KNN_5.predict(noclass_test),average='weighted'))
        KNN5_fim = time.time()
        KNN_5_tempo.append(KNN5_fim - KNN5_inicio)



        KNN7_inicio = time.time()
        KNN_7.fit(noclass_train, class_train)
        KNN_7_acerto.append(KNN_3.score(noclass_test, class_test))
        KNN_7_precision.append(precision_score(class_test,KNN_7.predict(noclass_test),average='weighted'))
        KNN_7_recall.append(recall_score(class_test,KNN_7.predict(noclass_test),average='weighted'))
        KNN_7_fscore.append(f1_score(class_test,KNN_7.predict(noclass_test),average='weighted'))
        KNN7_fim = time.time()
        KNN_7_tempo.append(KNN3_fim - KNN3_inicio)

        KNNP3_inicio = time.time()
        KNNP_3.fit(noclass_train, class_train)
        KNNP_3_acerto.append(KNNP_3.score(noclass_test, class_test))
        KNNP_3_precision.append(precision_score(class_test,KNNP_3.predict(noclass_test),average='weighted'))
        KNNP_3_recall.append(recall_score(class_test,KNNP_3.predict(noclass_test),average='weighted'))
        KNNP_3_fscore.append(f1_score(class_test,KNNP_3.predict(noclass_test),average='weighted'))
        KNNP3_fim = time.time()
        KNNP_3_tempo.append(KNNP3_fim - KNNP3_inicio)

        KNNP5_inicio = time.time()
        KNNP_5.fit(noclass_train, class_train)
        KNNP_5_acerto.append(KNNP_5.score(noclass_test, class_test))
        KNNP_5_precision.append(precision_score(class_test,KNNP_5.predict(noclass_test),average='weighted'))
        KNNP_5_recall.append(recall_score(class_test,KNNP_5.predict(noclass_test),average='weighted'))
        KNNP_5_fscore.append(f1_score(class_test,KNNP_5.predict(noclass_test),average='weighted'))
        KNNP5_fim = time.time()
        KNNP_5_tempo.append(KNNP5_fim - KNNP5_inicio)

        KNNP7_inicio = time.time()
        KNNP_7.fit(noclass_train, class_train)
        KNNP_7_acerto.append(KNNP_7.score(noclass_test, class_test))
        KNNP_7_precision.append(precision_score(class_test,KNNP_7.predict(noclass_test),average='weighted'))
        KNNP_7_recall.append(recall_score(class_test,KNNP_7.predict(noclass_test),average='weighted'))
        KNNP_7_fscore.append(f1_score(class_test,KNNP_7.predict(noclass_test),average='weighted'))
        KNNP7_fim = time.time()
        KNNP_7_tempo.append(KNNP7_fim - KNNP7_inicio)

        NaiveBayes_inicio = time.time()
        NaiveBayes.fit(noclass_train, class_train)
        NaiveBayes_acerto.append(NaiveBayes.score(noclass_test, class_test))
        NaiveBayes_precision.append(precision_score(class_test,NaiveBayes.predict(noclass_test),average='weighted'))
        NaiveBayes_recall.append(recall_score(class_test,NaiveBayes.predict(noclass_test),average='weighted'))
        NaiveBayes_fscore.append(f1_score(class_test,NaiveBayes.predict(noclass_test),average='weighted'))
        NaiveBayes_fim = time.time()
        NaiveBayes_tempo.append(NaiveBayes_fim - NaiveBayes_inicio)

        DecisionTree_inicio = time.time()
        DecisionTree.fit(noclass_train, class_train)
        DecisionTree_acerto.append(DecisionTree.score(noclass_test, class_test))
        DecisionTree_precision.append(precision_score(class_test,DecisionTree.predict(noclass_test),average='weighted'))
        DecisionTree_recall.append(recall_score(class_test,DecisionTree.predict(noclass_test),average='weighted'))
        DecisionTree_fscore.append(f1_score(class_test,DecisionTree.predict(noclass_test),average='weighted'))
        DecisionTree_fim = time.time()
        DecisionTree_tempo.append(DecisionTree_fim - DecisionTree_inicio)

        SVMachine_L_inicio = time.time()
        SVMachine_L.fit(noclass_train, class_train)
        SVMachine_L_acerto.append(SVMachine_L.score(noclass_test, class_test))
        SVMachine_L_precision.append(precision_score(class_test,SVMachine_L.predict(noclass_test),average='weighted'))
        SVMachine_L_recall.append(recall_score(class_test,SVMachine_L.predict(noclass_test),average='weighted'))
        SVMachine_L_fscore.append(f1_score(class_test,SVMachine_L.predict(noclass_test),average='weighted'))
        SVMachine_L_fim = time.time()
        SVMachine_L_tempo.append(SVMachine_L_fim - SVMachine_L_inicio)

        SVMachine_RBF_inicio = time.time()
        SVMachine_RBF.fit(noclass_train, class_train)
        SVMachine_RBF_acerto.append(SVMachine_RBF.score(noclass_test, class_test))
        SVMachine_RBF_recall.append(recall_score(class_test,SVMachine_RBF.predict(noclass_test),average='weighted'))
        SVMachine_RBF_precision.append(precision_score(class_test,SVMachine_RBF.predict(noclass_test),average='weighted'))
        SVMachine_RBF_fscore.append(f1_score(class_test,SVMachine_RBF.predict(noclass_test),average='weighted'))
        SVMachine_RBF_fim = time.time()
        SVMachine_RBF_tempo.append(SVMachine_RBF_fim - SVMachine_RBF_inicio)

        RegrLogistica_inicio = time.time()
        RegrLogistica.fit(noclass_train, class_train)
        RegrLogistica_acerto.append(RegrLogistica.score(noclass_test, class_test))
        RegreLogistica_precision.append(precision_score(class_test, (RegrLogistica.predict(noclass_test)),average='weighted'))
        RegrLogistica_recall.append(recall_score(class_test,RegrLogistica.predict(noclass_test),average='weighted'))
        RegrLogistica_fscore.append(f1_score(class_test,RegrLogistica.predict(noclass_test),average='weighted'))
        RegrLogistica_fim = time.time()
        RegrLogistica_tempo.append(RegrLogistica_fim - RegrLogistica_inicio)

        MLP_inicio = time.time()
        MLP.fit(noclass_train, class_train)
        MLP_acerto.append(MLP.score(noclass_test, class_test))
        MLPP_precision.append(precision_score(class_test,MLP.predict(noclass_test),average='weighted'))
        MLP_recall.append(recall_score(class_test,MLP.predict(noclass_test),average='weighted'))
        MLP_fscore.append(f1_score(class_test,MLP.predict(noclass_test),average='weighted'))
        MLP_fim = time.time()
        MLP_tempo.append(MLP_fim - MLP_inicio)

        RandomForest_inicio = time.time()
        RandomForest.fit(noclass_train, class_train)
        RandomForest_acerto.append(RandomForest.score(noclass_test, class_test))
        RandomForest_recall.append(recall_score(class_test,RandomForest.predict(noclass_test),average='weighted'))
        RandomForest_precision.append(precision_score(class_test,RandomForest.predict(noclass_test),average='weighted'))
        RandomForest_fscore.append(f1_score(class_test,RandomForest.predict(noclass_test),average='weighted'))
        RandomForest_fim = time.time()
        RandomForest_tempo.append(RandomForest_fim - RandomForest_inicio)


    dataFrame64 = dataFrame64.sample(frac=1)
    print("Terminou a ", x)
    tempo2 = time.time()
    print("Tempo da rodada ", x, (tempo2 - tempo1) / 60)

tempofinal = time.time()

fun.tendencia_central('KNN k = 3', KNN_3_acerto, KNN_3_tempo)
fun.tendencia_central('KNN k = 5', KNN_5_acerto, KNN_5_tempo)
fun.tendencia_central('KNN k = 7', KNN_7_acerto, KNN_7_tempo)
fun.tendencia_central('KNN Ponderado k = 3', KNNP_3_acerto, KNNP_3_tempo)
fun.tendencia_central('KNN Ponderado k = 5', KNNP_5_acerto, KNNP_5_tempo)
fun.tendencia_central('KNN Ponderado k = 7', KNNP_7_acerto, KNNP_7_tempo)
fun.tendencia_central('Naïve Bayes', NaiveBayes_acerto, NaiveBayes_tempo)
fun.tendencia_central('Árvore de decisão', DecisionTree_acerto, DecisionTree_tempo)
fun.tendencia_central('MLP', MLP_acerto, MLP_tempo)
fun.tendencia_central('Regressão Logística', RegrLogistica_acerto, RegrLogistica_tempo)
fun.tendencia_central('SVM linear', SVMachine_L_acerto, SVMachine_L_tempo)
fun.tendencia_central('SVM RBF', SVMachine_RBF_acerto, SVMachine_RBF_tempo)
fun.tendencia_central('Random Forest', RandomForest_acerto,RandomForest_tempo)

mediasacuracias = {


    "KNN Ponderado k = 3": np.mean(KNNP_3_acerto),
    "KNN Ponderado k = 5": np.mean(KNNP_5_acerto),
    "KNN Ponderado k = 7": np.mean(KNNP_7_acerto),
    "Naive Bayes": np.mean(NaiveBayes_acerto),
    "KNN k = 3": np.mean(KNN_3_acerto),
    "KNN k = 5": np.mean(KNN_5_acerto),
    "KNN k = 7": np.mean(KNN_7_acerto),
    "Decision Tree": np.mean(DecisionTree_acerto),
    "SVM Linear": np.mean(SVMachine_L_acerto),
    "SVM RBF": np.mean(SVMachine_RBF_acerto),
    "Regressao Logistica": np.mean(RegrLogistica_acerto),
    "MLP": np.mean(MLP_acerto),
    "Random Forest": np.mean(RandomForest_acerto)
}

mediasacuracias = sorted(mediasacuracias.items(),
                         key=lambda x: x[1])
print(mediasacuracias)
print("Tempo total: ", (tempofinal - tempoinicial) / 60)
Acuracia64 = [KNN_3_acerto,KNN_5_acerto,KNN_7_acerto,KNNP_3_acerto,KNNP_5_acerto,KNNP_7_acerto,
               NaiveBayes_acerto,DecisionTree_acerto,MLP_acerto,RegrLogistica_acerto,SVMachine_L_acerto,
               SVMachine_RBF_acerto,RandomForest_acerto]
Precision64 = [KNN_3_precision,KNN_5_precision,KNN_7_precision,KNNP_3_precision,KNNP_5_precision,KNNP_7_precision,
         NaiveBayes_precision,DecisionTree_precision,MLPP_precision,RegreLogistica_precision,SVMachine_L_precision,
         SVMachine_RBF_precision,RandomForest_precision]
Recall64 = [KNN_3_recall,KNN_5_recall,KNN_7_recall,KNNP_3_recall,KNNP_5_recall,KNNP_7_recall,NaiveBayes_recall,
          DecisionTree_recall,MLP_recall,RegrLogistica_recall,SVMachine_L_recall,SVMachine_RBF_recall,RandomForest_recall]

Fscore64 = [KNN_3_fscore,KNN_5_fscore,KNN_7_fscore,KNNP_3_fscore,KNN_5_fscore,KNN_7_fscore,NaiveBayes_fscore,DecisionTree_fscore,
          MLP_fscore,RegrLogistica_fscore,SVMachine_L_fscore, SVMachine_RBF_fscore,RandomForest_fscore]

Acuracia64 = pd.DataFrame(Acuracia64)
Precision64 = pd.DataFrame(Precision64)
Recall64 = pd.DataFrame(Recall64)
Fscore64 = pd.DataFrame(Fscore64)

Acuracia64.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Acuracia64.csv')
Precision64.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Precision64.csv')
Recall64.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Recall64.csv')
Fscore64.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Fscore64.csv')




# USANDO BASE DE 32x32



print('comecou o K fold')

tempoinicial = time.time()

for x in range(0, 10):

    tempo1 = time.time()
    cols = list(dataFrame32.columns)
    cols.remove('Classe')
    df_images_noclass = dataFrame32[cols]
    df_images_class = dataFrame32['Classe']
    c = kfold.split(dataFrame32)

    for train_index, test_index in c:

        noclass_train, noclass_test = df_images_noclass.iloc[train_index], df_images_noclass.iloc[test_index]
        class_train, class_test = df_images_class.iloc[train_index], df_images_class.iloc[test_index]

        KNN3_inicio = time.time()
        KNN_3.fit(noclass_train, class_train)
        KNN_3_acerto.append(KNN_3.score(noclass_test, class_test))
        KNN_3_recall. append(recall_score(class_test, KNN_3.predict(noclass_test),average='weighted'))
        KNN_3_precision.append(precision_score(class_test,KNN_3.predict(noclass_test),average='weighted'))
        KNN_3_fscore.append(f1_score(class_test,KNN_3.predict(noclass_test),average='weighted'))
        KNN3_fim = time.time()
        KNN_3_tempo.append(KNN3_fim - KNN3_inicio)

        KNN5_inicio = time.time()
        KNN_5.fit(noclass_train, class_train)
        KNN_5_acerto.append(KNN_5.score(noclass_test, class_test))
        KNN_5_recall.append(recall_score(class_test,KNN_5.predict(noclass_test),average='weighted'))
        KNN_5_precision.append(precision_score(class_test,KNN_5.predict(noclass_test),average='weighted'))
        KNN_5_fscore.append(f1_score(class_test,KNN_5.predict(noclass_test),average='weighted'))
        KNN5_fim = time.time()
        KNN_5_tempo.append(KNN5_fim - KNN5_inicio)



        KNN7_inicio = time.time()
        KNN_7.fit(noclass_train, class_train)
        KNN_7_acerto.append(KNN_3.score(noclass_test, class_test))
        KNN_7_precision.append(precision_score(class_test,KNN_7.predict(noclass_test),average='weighted'))
        KNN_7_recall.append(recall_score(class_test,KNN_7.predict(noclass_test),average='weighted'))
        KNN_7_fscore.append(f1_score(class_test,KNN_7.predict(noclass_test),average='weighted'))
        KNN7_fim = time.time()
        KNN_7_tempo.append(KNN3_fim - KNN3_inicio)

        KNNP3_inicio = time.time()
        KNNP_3.fit(noclass_train, class_train)
        KNNP_3_acerto.append(KNNP_3.score(noclass_test, class_test))
        KNNP_3_precision.append(precision_score(class_test,KNNP_3.predict(noclass_test),average='weighted'))
        KNNP_3_recall.append(recall_score(class_test,KNNP_3.predict(noclass_test),average='weighted'))
        KNNP_3_fscore.append(f1_score(class_test,KNNP_3.predict(noclass_test),average='weighted'))
        KNNP3_fim = time.time()
        KNNP_3_tempo.append(KNNP3_fim - KNNP3_inicio)

        KNNP5_inicio = time.time()
        KNNP_5.fit(noclass_train, class_train)
        KNNP_5_acerto.append(KNNP_5.score(noclass_test, class_test))
        KNNP_5_precision.append(precision_score(class_test,KNNP_5.predict(noclass_test),average='weighted'))
        KNNP_5_recall.append(recall_score(class_test,KNNP_5.predict(noclass_test),average='weighted'))
        KNNP_5_fscore.append(f1_score(class_test,KNNP_5.predict(noclass_test),average='weighted'))
        KNNP5_fim = time.time()
        KNNP_5_tempo.append(KNNP5_fim - KNNP5_inicio)

        KNNP7_inicio = time.time()
        KNNP_7.fit(noclass_train, class_train)
        KNNP_7_acerto.append(KNNP_7.score(noclass_test, class_test))
        KNNP_7_precision.append(precision_score(class_test,KNNP_7.predict(noclass_test),average='weighted'))
        KNNP_7_recall.append(recall_score(class_test,KNNP_7.predict(noclass_test),average='weighted'))
        KNNP_7_fscore.append(f1_score(class_test,KNNP_7.predict(noclass_test),average='weighted'))
        KNNP7_fim = time.time()
        KNNP_7_tempo.append(KNNP7_fim - KNNP7_inicio)

        NaiveBayes_inicio = time.time()
        NaiveBayes.fit(noclass_train, class_train)
        NaiveBayes_acerto.append(NaiveBayes.score(noclass_test, class_test))
        NaiveBayes_precision.append(precision_score(class_test,NaiveBayes.predict(noclass_test),average='weighted'))
        NaiveBayes_recall.append(recall_score(class_test,NaiveBayes.predict(noclass_test),average='weighted'))
        NaiveBayes_fscore.append(f1_score(class_test,NaiveBayes.predict(noclass_test),average='weighted'))
        NaiveBayes_fim = time.time()
        NaiveBayes_tempo.append(NaiveBayes_fim - NaiveBayes_inicio)

        DecisionTree_inicio = time.time()
        DecisionTree.fit(noclass_train, class_train)
        DecisionTree_acerto.append(DecisionTree.score(noclass_test, class_test))
        DecisionTree_precision.append(precision_score(class_test,DecisionTree.predict(noclass_test),average='weighted'))
        DecisionTree_recall.append(recall_score(class_test,DecisionTree.predict(noclass_test),average='weighted'))
        DecisionTree_fscore.append(f1_score(class_test,DecisionTree.predict(noclass_test),average='weighted'))
        DecisionTree_fim = time.time()
        DecisionTree_tempo.append(DecisionTree_fim - DecisionTree_inicio)

        SVMachine_L_inicio = time.time()
        SVMachine_L.fit(noclass_train, class_train)
        SVMachine_L_acerto.append(SVMachine_L.score(noclass_test, class_test))
        SVMachine_L_precision.append(precision_score(class_test,SVMachine_L.predict(noclass_test),average='weighted'))
        SVMachine_L_recall.append(recall_score(class_test,SVMachine_L.predict(noclass_test),average='weighted'))
        SVMachine_L_fscore.append(f1_score(class_test,SVMachine_L.predict(noclass_test),average='weighted'))
        SVMachine_L_fim = time.time()
        SVMachine_L_tempo.append(SVMachine_L_fim - SVMachine_L_inicio)

        SVMachine_RBF_inicio = time.time()
        SVMachine_RBF.fit(noclass_train, class_train)
        SVMachine_RBF_acerto.append(SVMachine_RBF.score(noclass_test, class_test))
        SVMachine_RBF_recall.append(recall_score(class_test,SVMachine_RBF.predict(noclass_test),average='weighted'))
        SVMachine_RBF_precision.append(precision_score(class_test,SVMachine_RBF.predict(noclass_test),average='weighted'))
        SVMachine_RBF_fscore.append(f1_score(class_test,SVMachine_RBF.predict(noclass_test),average='weighted'))
        SVMachine_RBF_fim = time.time()
        SVMachine_RBF_tempo.append(SVMachine_RBF_fim - SVMachine_RBF_inicio)

        RegrLogistica_inicio = time.time()
        RegrLogistica.fit(noclass_train, class_train)
        RegrLogistica_acerto.append(RegrLogistica.score(noclass_test, class_test))
        RegreLogistica_precision.append(precision_score(class_test, (RegrLogistica.predict(noclass_test)),average='weighted'))
        RegrLogistica_recall.append(recall_score(class_test,RegrLogistica.predict(noclass_test),average='weighted'))
        RegrLogistica_fscore.append(f1_score(class_test,RegrLogistica.predict(noclass_test),average='weighted'))
        RegrLogistica_fim = time.time()
        RegrLogistica_tempo.append(RegrLogistica_fim - RegrLogistica_inicio)

        MLP_inicio = time.time()
        MLP.fit(noclass_train, class_train)
        MLP_acerto.append(MLP.score(noclass_test, class_test))
        MLPP_precision.append(precision_score(class_test,MLP.predict(noclass_test),average='weighted'))
        MLP_recall.append(recall_score(class_test,MLP.predict(noclass_test),average='weighted'))
        MLP_fscore.append(f1_score(class_test,MLP.predict(noclass_test),average='weighted'))
        MLP_fim = time.time()
        MLP_tempo.append(MLP_fim - MLP_inicio)

        RandomForest_inicio = time.time()
        RandomForest.fit(noclass_train, class_train)
        RandomForest_acerto.append(RandomForest.score(noclass_test, class_test))
        RandomForest_recall.append(recall_score(class_test,RandomForest.predict(noclass_test),average='weighted'))
        RandomForest_precision.append(precision_score(class_test,RandomForest.predict(noclass_test),average='weighted'))
        RandomForest_fscore.append(f1_score(class_test,RandomForest.predict(noclass_test),average='weighted'))
        RandomForest_fim = time.time()
        RandomForest_tempo.append(RandomForest_fim - RandomForest_inicio)


    dataFrame32 = dataFrame32.sample(frac=1)
    print("Terminou a ", x)
    tempo2 = time.time()
    print("Tempo da rodada ", x, (tempo2 - tempo1) / 60)

tempofinal = time.time()

fun.tendencia_central('KNN k = 3', KNN_3_acerto, KNN_3_tempo)
fun.tendencia_central('KNN k = 5', KNN_5_acerto, KNN_5_tempo)
fun.tendencia_central('KNN k = 7', KNN_7_acerto, KNN_7_tempo)
fun.tendencia_central('KNN Ponderado k = 3', KNNP_3_acerto, KNNP_3_tempo)
fun.tendencia_central('KNN Ponderado k = 5', KNNP_5_acerto, KNNP_5_tempo)
fun.tendencia_central('KNN Ponderado k = 7', KNNP_7_acerto, KNNP_7_tempo)
fun.tendencia_central('Naïve Bayes', NaiveBayes_acerto, NaiveBayes_tempo)
fun.tendencia_central('Árvore de decisão', DecisionTree_acerto, DecisionTree_tempo)
fun.tendencia_central('MLP', MLP_acerto, MLP_tempo)
fun.tendencia_central('Regressão Logística', RegrLogistica_acerto, RegrLogistica_tempo)
fun.tendencia_central('SVM linear', SVMachine_L_acerto, SVMachine_L_tempo)
fun.tendencia_central('SVM RBF', SVMachine_RBF_acerto, SVMachine_RBF_tempo)
fun.tendencia_central('Random Forest', RandomForest_acerto,RandomForest_tempo)

mediasacuracias = {


    "KNN Ponderado k = 3": np.mean(KNNP_3_acerto),
    "KNN Ponderado k = 5": np.mean(KNNP_5_acerto),
    "KNN Ponderado k = 7": np.mean(KNNP_7_acerto),
    "Naive Bayes": np.mean(NaiveBayes_acerto),
    "KNN k = 3": np.mean(KNN_3_acerto),
    "KNN k = 5": np.mean(KNN_5_acerto),
    "KNN k = 7": np.mean(KNN_7_acerto),
    "Decision Tree": np.mean(DecisionTree_acerto),
    "SVM Linear": np.mean(SVMachine_L_acerto),
    "SVM RBF": np.mean(SVMachine_RBF_acerto),
    "Regressao Logistica": np.mean(RegrLogistica_acerto),
    "MLP": np.mean(MLP_acerto),
    "Random Forest": np.mean(RandomForest_acerto)
}

mediasacuracias = sorted(mediasacuracias.items(),
                         key=lambda x: x[1])
print(mediasacuracias)
print("Tempo total: ", (tempofinal - tempoinicial) / 60)
Acuracia32 = [KNN_3_acerto,KNN_5_acerto,KNN_7_acerto,KNNP_3_acerto,KNNP_5_acerto,KNNP_7_acerto,
               NaiveBayes_acerto,DecisionTree_acerto,MLP_acerto,RegrLogistica_acerto,SVMachine_L_acerto,
               SVMachine_RBF_acerto,RandomForest_acerto]
Precision32 = [KNN_3_precision,KNN_5_precision,KNN_7_precision,KNNP_3_precision,KNNP_5_precision,KNNP_7_precision,
         NaiveBayes_precision,DecisionTree_precision,MLPP_precision,RegreLogistica_precision,SVMachine_L_precision,
         SVMachine_RBF_precision,RandomForest_precision]
Recall32 = [KNN_3_recall,KNN_5_recall,KNN_7_recall,KNNP_3_recall,KNNP_5_recall,KNNP_7_recall,NaiveBayes_recall,
          DecisionTree_recall,MLP_recall,RegrLogistica_recall,SVMachine_L_recall,SVMachine_RBF_recall,RandomForest_recall]

Fscore32 = [KNN_3_fscore,KNN_5_fscore,KNN_7_fscore,KNNP_3_fscore,KNN_5_fscore,KNN_7_fscore,NaiveBayes_fscore,DecisionTree_fscore,
          MLP_fscore,RegrLogistica_fscore,SVMachine_L_fscore, SVMachine_RBF_fscore,RandomForest_fscore]

Acuracia32 = pd.DataFrame(Acuracia32)
Precision32 = pd.DataFrame(Precision32)
Recall32 = pd.DataFrame(Recall32)
Fscore32 = pd.DataFrame(Fscore32)

Acuracia32.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Acuracia32.csv')
Precision32.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Precision32.csv')
Recall32.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Recall32.csv')
Fscore32.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Fscore32.csv')


'''

'''

# USANDO A BASE COM IMAGENS 16x16

print('comecou o K fold')

tempoinicial = time.time()

for x in range(0, 10):

    tempo1 = time.time()
    cols = list(dataFrame16.columns)
    cols.remove('Classe')
    df_images_noclass = dataFrame16[cols]
    df_images_class = dataFrame16['Classe']
    c = kfold.split(dataFrame16)

    for train_index, test_index in c:

        noclass_train, noclass_test = df_images_noclass.iloc[train_index], df_images_noclass.iloc[test_index]
        class_train, class_test = df_images_class.iloc[train_index], df_images_class.iloc[test_index]

        KNN3_inicio = time.time()
        KNN_3.fit(noclass_train, class_train)
        KNN_3_acerto.append(KNN_3.score(noclass_test, class_test))
        KNN_3_recall. append(recall_score(class_test, KNN_3.predict(noclass_test),average='weighted'))
        KNN_3_precision.append(precision_score(class_test,KNN_3.predict(noclass_test),average='weighted'))
        KNN_3_fscore.append(f1_score(class_test,KNN_3.predict(noclass_test),average='weighted'))
        KNN3_fim = time.time()
        KNN_3_tempo.append(KNN3_fim - KNN3_inicio)

        KNN5_inicio = time.time()
        KNN_5.fit(noclass_train, class_train)
        KNN_5_acerto.append(KNN_5.score(noclass_test, class_test))
        KNN_5_recall.append(recall_score(class_test,KNN_5.predict(noclass_test),average='weighted'))
        KNN_5_precision.append(precision_score(class_test,KNN_5.predict(noclass_test),average='weighted'))
        KNN_5_fscore.append(f1_score(class_test,KNN_5.predict(noclass_test),average='weighted'))
        KNN5_fim = time.time()
        KNN_5_tempo.append(KNN5_fim - KNN5_inicio)


        KNN7_inicio = time.time()
        KNN_7.fit(noclass_train, class_train)
        KNN_7_acerto.append(KNN_3.score(noclass_test, class_test))
        KNN_7_precision.append(precision_score(class_test,KNN_7.predict(noclass_test),average='weighted'))
        KNN_7_recall.append(recall_score(class_test,KNN_7.predict(noclass_test),average='weighted'))
        KNN_7_fscore.append(f1_score(class_test,KNN_7.predict(noclass_test),average='weighted'))
        KNN7_fim = time.time()
        KNN_7_tempo.append(KNN3_fim - KNN3_inicio)

        KNNP3_inicio = time.time()
        KNNP_3.fit(noclass_train, class_train)
        KNNP_3_acerto.append(KNNP_3.score(noclass_test, class_test))
        KNNP_3_precision.append(precision_score(class_test,KNNP_3.predict(noclass_test),average='weighted'))
        KNNP_3_recall.append(recall_score(class_test,KNNP_3.predict(noclass_test),average='weighted'))
        KNNP_3_fscore.append(f1_score(class_test,KNNP_3.predict(noclass_test),average='weighted'))
        KNNP3_fim = time.time()
        KNNP_3_tempo.append(KNNP3_fim - KNNP3_inicio)

        KNNP5_inicio = time.time()
        KNNP_5.fit(noclass_train, class_train)
        KNNP_5_acerto.append(KNNP_5.score(noclass_test, class_test))
        KNNP_5_precision.append(precision_score(class_test,KNNP_5.predict(noclass_test),average='weighted'))
        KNNP_5_recall.append(recall_score(class_test,KNNP_5.predict(noclass_test),average='weighted'))
        KNNP_5_fscore.append(f1_score(class_test,KNNP_5.predict(noclass_test),average='weighted'))
        KNNP5_fim = time.time()
        KNNP_5_tempo.append(KNNP5_fim - KNNP5_inicio)

        KNNP7_inicio = time.time()
        KNNP_7.fit(noclass_train, class_train)
        KNNP_7_acerto.append(KNNP_7.score(noclass_test, class_test))
        KNNP_7_precision.append(precision_score(class_test,KNNP_7.predict(noclass_test),average='weighted'))
        KNNP_7_recall.append(recall_score(class_test,KNNP_7.predict(noclass_test),average='weighted'))
        KNNP_7_fscore.append(f1_score(class_test,KNNP_7.predict(noclass_test),average='weighted'))
        KNNP7_fim = time.time()
        KNNP_7_tempo.append(KNNP7_fim - KNNP7_inicio)

        NaiveBayes_inicio = time.time()
        NaiveBayes.fit(noclass_train, class_train)
        NaiveBayes_acerto.append(NaiveBayes.score(noclass_test, class_test))
        NaiveBayes_precision.append(precision_score(class_test,NaiveBayes.predict(noclass_test),average='weighted'))
        NaiveBayes_recall.append(recall_score(class_test,NaiveBayes.predict(noclass_test),average='weighted'))
        NaiveBayes_fscore.append(f1_score(class_test,NaiveBayes.predict(noclass_test),average='weighted'))
        NaiveBayes_fim = time.time()
        NaiveBayes_tempo.append(NaiveBayes_fim - NaiveBayes_inicio)

        DecisionTree_inicio = time.time()
        DecisionTree.fit(noclass_train, class_train)
        DecisionTree_acerto.append(DecisionTree.score(noclass_test, class_test))
        DecisionTree_precision.append(precision_score(class_test,DecisionTree.predict(noclass_test),average='weighted'))
        DecisionTree_recall.append(recall_score(class_test,DecisionTree.predict(noclass_test),average='weighted'))
        DecisionTree_fscore.append(f1_score(class_test,DecisionTree.predict(noclass_test),average='weighted'))
        DecisionTree_fim = time.time()
        DecisionTree_tempo.append(DecisionTree_fim - DecisionTree_inicio)

        SVMachine_L_inicio = time.time()
        SVMachine_L.fit(noclass_train, class_train)
        SVMachine_L_acerto.append(SVMachine_L.score(noclass_test, class_test))
        SVMachine_L_precision.append(precision_score(class_test,SVMachine_L.predict(noclass_test),average='weighted'))
        SVMachine_L_recall.append(recall_score(class_test,SVMachine_L.predict(noclass_test),average='weighted'))
        SVMachine_L_fscore.append(f1_score(class_test,SVMachine_L.predict(noclass_test),average='weighted'))
        SVMachine_L_fim = time.time()
        SVMachine_L_tempo.append(SVMachine_L_fim - SVMachine_L_inicio)

        SVMachine_RBF_inicio = time.time()
        SVMachine_RBF.fit(noclass_train, class_train)
        SVMachine_RBF_acerto.append(SVMachine_RBF.score(noclass_test, class_test))
        SVMachine_RBF_recall.append(recall_score(class_test,SVMachine_RBF.predict(noclass_test),average='weighted'))
        SVMachine_RBF_precision.append(precision_score(class_test,SVMachine_RBF.predict(noclass_test),average='weighted'))
        SVMachine_RBF_fscore.append(f1_score(class_test,SVMachine_RBF.predict(noclass_test),average='weighted'))
        SVMachine_RBF_fim = time.time()
        SVMachine_RBF_tempo.append(SVMachine_RBF_fim - SVMachine_RBF_inicio)

        RegrLogistica_inicio = time.time()
        RegrLogistica.fit(noclass_train, class_train)
        RegrLogistica_acerto.append(RegrLogistica.score(noclass_test, class_test))
        RegreLogistica_precision.append(precision_score(class_test, (RegrLogistica.predict(noclass_test)),average='weighted'))
        RegrLogistica_recall.append(recall_score(class_test,RegrLogistica.predict(noclass_test),average='weighted'))
        RegrLogistica_fscore.append(f1_score(class_test,RegrLogistica.predict(noclass_test),average='weighted'))
        RegrLogistica_fim = time.time()
        RegrLogistica_tempo.append(RegrLogistica_fim - RegrLogistica_inicio)

        MLP_inicio = time.time()
        MLP.fit(noclass_train, class_train)
        MLP_acerto.append(MLP.score(noclass_test, class_test))
        MLPP_precision.append(precision_score(class_test,MLP.predict(noclass_test),average='weighted'))
        MLP_recall.append(recall_score(class_test,MLP.predict(noclass_test),average='weighted'))
        MLP_fscore.append(f1_score(class_test,MLP.predict(noclass_test),average='weighted'))
        MLP_fim = time.time()
        MLP_tempo.append(MLP_fim - MLP_inicio)

        RandomForest_inicio = time.time()
        RandomForest.fit(noclass_train, class_train)
        RandomForest_acerto.append(RandomForest.score(noclass_test, class_test))
        RandomForest_recall.append(recall_score(class_test,RandomForest.predict(noclass_test),average='weighted'))
        RandomForest_precision.append(precision_score(class_test,RandomForest.predict(noclass_test),average='weighted'))
        RandomForest_fscore.append(f1_score(class_test,RandomForest.predict(noclass_test),average='weighted'))
        RandomForest_fim = time.time()
        RandomForest_tempo.append(RandomForest_fim - RandomForest_inicio)


    dataFrame16 = dataFrame16.sample(frac=1)
    print("Terminou a ", x)
    tempo2 = time.time()
    print("Tempo da rodada ", x, (tempo2 - tempo1) / 60)

tempofinal = time.time()

fun.tendencia_central('KNN k = 3', KNN_3_acerto, KNN_3_tempo)
fun.tendencia_central('KNN k = 5', KNN_5_acerto, KNN_5_tempo)
fun.tendencia_central('KNN k = 7', KNN_7_acerto, KNN_7_tempo)
fun.tendencia_central('KNN Ponderado k = 3', KNNP_3_acerto, KNNP_3_tempo)
fun.tendencia_central('KNN Ponderado k = 5', KNNP_5_acerto, KNNP_5_tempo)
fun.tendencia_central('KNN Ponderado k = 7', KNNP_7_acerto, KNNP_7_tempo)
fun.tendencia_central('Naïve Bayes', NaiveBayes_acerto, NaiveBayes_tempo)
fun.tendencia_central('Árvore de decisão', DecisionTree_acerto, DecisionTree_tempo)
fun.tendencia_central('MLP', MLP_acerto, MLP_tempo)
fun.tendencia_central('Regressão Logística', RegrLogistica_acerto, RegrLogistica_tempo)
fun.tendencia_central('SVM linear', SVMachine_L_acerto, SVMachine_L_tempo)
fun.tendencia_central('SVM RBF', SVMachine_RBF_acerto, SVMachine_RBF_tempo)
fun.tendencia_central('Random Forest', RandomForest_acerto,RandomForest_tempo)

mediasacuracias = {


    "KNN Ponderado k = 3": np.mean(KNNP_3_acerto),
    "KNN Ponderado k = 5": np.mean(KNNP_5_acerto),
    "KNN Ponderado k = 7": np.mean(KNNP_7_acerto),
    "Naive Bayes": np.mean(NaiveBayes_acerto),
    "KNN k = 3": np.mean(KNN_3_acerto),
    "KNN k = 5": np.mean(KNN_5_acerto),
    "KNN k = 7": np.mean(KNN_7_acerto),
    "Decision Tree": np.mean(DecisionTree_acerto),
    "SVM Linear": np.mean(SVMachine_L_acerto),
    "SVM RBF": np.mean(SVMachine_RBF_acerto),
    "Regressao Logistica": np.mean(RegrLogistica_acerto),
    "MLP": np.mean(MLP_acerto),
    "Random Forest": np.mean(RandomForest_acerto)
}

mediasacuracias = sorted(mediasacuracias.items(),
                         key=lambda x: x[1])
print(mediasacuracias)
print("Tempo total: ", (tempofinal - tempoinicial) / 60)
Acuracia16 = [KNN_3_acerto,KNN_5_acerto,KNN_7_acerto,KNNP_3_acerto,KNNP_5_acerto,KNNP_7_acerto,
               NaiveBayes_acerto,DecisionTree_acerto,MLP_acerto,RegrLogistica_acerto,SVMachine_L_acerto,
               SVMachine_RBF_acerto,RandomForest_acerto]
Precision16 = [KNN_3_precision,KNN_5_precision,KNN_7_precision,KNNP_3_precision,KNNP_5_precision,KNNP_7_precision,
         NaiveBayes_precision,DecisionTree_precision,MLPP_precision,RegreLogistica_precision,SVMachine_L_precision,
         SVMachine_RBF_precision,RandomForest_precision]
Recall16 = [KNN_3_recall,KNN_5_recall,KNN_7_recall,KNNP_3_recall,KNNP_5_recall,KNNP_7_recall,NaiveBayes_recall,
          DecisionTree_recall,MLP_recall,RegrLogistica_recall,SVMachine_L_recall,SVMachine_RBF_recall,RandomForest_recall]

Fscore16 = [KNN_3_fscore,KNN_5_fscore,KNN_7_fscore,KNNP_3_fscore,KNN_5_fscore,KNN_7_fscore,NaiveBayes_fscore,DecisionTree_fscore,
          MLP_fscore,RegrLogistica_fscore,SVMachine_L_fscore, SVMachine_RBF_fscore,RandomForest_fscore]

Acuracia16 = pd.DataFrame(Acuracia16)
Precision16 = pd.DataFrame(Precision16)
Recall16 = pd.DataFrame(Recall16)
Fscore16 = pd.DataFrame(Fscore16)

Acuracia16.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Acuracia16.csv')
Precision16.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Precision16.csv')
Recall16.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Recall16.csv')
Fscore16.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Fscore16.csv')
'''
