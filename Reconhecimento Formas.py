import pandas as pd
import cv2
import numpy as np
import FuncoesML as fun
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
circles = []
circles = fun.loadFiles('C:/Users/Auricelia/Desktop/DataSetsML/New_shapes_dataset/circles/*.jpg',circles)

ellipsis = []
ellipsis = fun.loadFiles('C:/Users/Auricelia/Desktop/DataSetsML/New_shapes_dataset/ellipses/*.jpg', ellipsis)

hexagons = []
hexagons = fun.loadFiles('C:/Users/Auricelia/Desktop/DataSetsML/New_shapes_dataset/hexagons/*.jpg', hexagons)

lines = []
lines = fun.loadFiles('C:/Users/Auricelia/Desktop/DataSetsML/New_shapes_dataset/lines/*.jpg', lines)

rectangles = []
rectangles = fun.loadFiles('C:/Users/Auricelia/Desktop/DataSetsML/New_shapes_dataset/rectangles/*.jpg', rectangles)

rhombuses = []
rhombuses = fun.loadFiles('C:/Users/Auricelia/Desktop/DataSetsML/New_shapes_dataset/rhombuses/*.jpg', rhombuses)

squares = []
squares = fun.loadFiles('C:/Users/Auricelia/Desktop/DataSetsML/New_shapes_dataset/squares/*.jpg', squares)

trapezia = []
trapezia = fun.loadFiles('C:/Users/Auricelia/Desktop/DataSetsML/New_shapes_dataset/trapezia/*.jpg', trapezia)

triangles = []
triangles = fun.loadFiles('C:/Users/Auricelia/Desktop/DataSetsML/New_shapes_dataset/triangles/*.jpg', triangles)
print('terminou load images')


# PRE PROCESSING

#criando cópias de cada uma das pastas para redimensionar as imagens
#quadrados

squares16 = squares.copy()
squares32 = squares.copy()
squares64 = squares.copy()
squares128 = squares.copy()

squares16 = fun.resizeImages(squares16,16,16)
squares32 = fun.resizeImages(squares32,32,32)
squares64 = fun.resizeImages(squares64,64,64)
squares128 = fun.resizeImages(squares128,128,128)

#círculos
circles16 = circles.copy()
circles32 = circles.copy()
circles64 = circles.copy()
circles128 = circles.copy()

circles16 = fun.resizeImages(circles16,16,16)
circles32 = fun.resizeImages(circles32,32,32)
circles64 = fun.resizeImages(circles64,64,64)
circles128 = fun.resizeImages(circles128,128,128)

#elipses
ellipsis16 = ellipsis.copy()
ellipsis32 = ellipsis.copy()
ellipsis64 = ellipsis.copy()
ellipsis128 = ellipsis.copy()

ellipsis16 = fun.resizeImages(ellipsis16,16,16)
ellipsis32 = fun.resizeImages(ellipsis32,32,32)
ellipsis64 = fun.resizeImages(ellipsis64,64,64)
ellipsis128 = fun.resizeImages(ellipsis128,128,128)

#hexágonos
hexagons16 = hexagons.copy()
hexagons32 = hexagons.copy()
hexagons64 = hexagons.copy()
hexagons128 = hexagons.copy()

hexagons16 = fun.resizeImages(hexagons16,16,16)
hexagons32 = fun.resizeImages(hexagons32,32,32)
hexagons64 = fun.resizeImages(hexagons64,64,64)
hexagons128 = fun.resizeImages(hexagons128,128,128)

#linhas
lines16 = lines.copy()
lines32 = lines.copy()
lines64 = lines.copy()
lines128 = lines.copy()

lines16 = fun.resizeImages(lines16,16,16)
lines32 = fun.resizeImages(lines32,32,32)
lines64 = fun.resizeImages(lines64,64,64)
lines128 = fun.resizeImages(lines128,128,128)

#retângulos
rectangles16 = rectangles.copy()
rectangles32 = rectangles.copy()
rectangles64 = rectangles.copy()
rectangles128 = rectangles.copy()

rectangles16 = fun.resizeImages(rectangles16,16,16)
rectangles32 = fun.resizeImages(rectangles32,32,32)
rectangles64 = fun.resizeImages(rectangles64,64,64)
rectangles128 = fun.resizeImages(rectangles128,128,128)

#losangos
rhombuses16 = rhombuses.copy()
rhombuses32 = rhombuses.copy()
rhombuses64 = rhombuses.copy()
rhombuses128 = rhombuses.copy()

rhombuses16 = fun.resizeImages(rhombuses16,16,16)
rhombuses32 = fun.resizeImages(rhombuses32,32,32)
rhombuses64 = fun.resizeImages(rhombuses64,64,64)
rhombuses128 = fun.resizeImages(rhombuses128,128,128)

#trapézios
trapezia16 = trapezia.copy()
trapezia32 = trapezia.copy()
trapezia64 = trapezia.copy()
trapezia128 = trapezia.copy()

trapezia16 = fun.resizeImages(trapezia16,16,16)
trapezia32 = fun.resizeImages(trapezia32,32,32)
trapezia64 = fun.resizeImages(trapezia64,64,64)
trapezia128 = fun.resizeImages(trapezia128,128,128)

#triângulos
triangles16 = triangles.copy()
triangles32 = triangles.copy()
triangles64 = triangles.copy()
triangles128 = triangles.copy()

triangles16 = fun.resizeImages(triangles16,16,16)
triangles32 = fun.resizeImages(triangles32,32,32)
triangles64 = fun.resizeImages(triangles64,64,64)
triangles128 = fun.resizeImages(triangles128,128,128)

#convertendo para níveis de cinza

squares16 = fun.grayConversion(squares16)
squares32 = fun.grayConversion(squares32)
squares64 = fun.grayConversion(squares64)
squares128 = fun.grayConversion(squares128)


circles16 = fun.grayConversion(circles16)
circles32 = fun.grayConversion(circles32)
circles64 = fun.grayConversion(circles64)
circles128 = fun.grayConversion(circles128)

triangles16 = fun.grayConversion(triangles16)
triangles32 = fun.grayConversion(triangles32)
triangles64 = fun.grayConversion(triangles64)
triangles128 = fun.grayConversion(triangles128)


trapezia16 = fun.grayConversion(trapezia16)
trapezia32 = fun.grayConversion(trapezia32)
trapezia64 = fun.grayConversion(trapezia64)
trapezia128 = fun.grayConversion(trapezia128)

rhombuses16 = fun.grayConversion(rhombuses16)
rhombuses32 = fun.grayConversion(rhombuses32)
rhombuses64 = fun.grayConversion(rhombuses64)
rhombuses128 = fun.grayConversion(rhombuses128)

rectangles16= fun.grayConversion(rectangles16)
rectangles32 = fun.grayConversion(rectangles32)
rectangles64 = fun.grayConversion(rectangles64)
rectangles128 = fun.grayConversion(rectangles128)

lines16 = fun.grayConversion(lines16)
lines32 = fun.grayConversion(lines32)
lines64 = fun.grayConversion(lines64)
lines128 = fun.grayConversion(lines128)

hexagons16 = fun.grayConversion(hexagons16)
hexagons32 = fun.grayConversion(hexagons32)
hexagons64 = fun.grayConversion(hexagons64)
hexagons128 = fun.grayConversion(hexagons128)

ellipsis16 = fun.grayConversion(ellipsis16)
ellipsis32 = fun.grayConversion(ellipsis32)
ellipsis64 = fun.grayConversion(ellipsis64)
ellipsis128 = fun.grayConversion(ellipsis128)

#aplicando o filtro gaussiano

squares16 = fun.blurConversion(squares16,5,0)
squares32 = fun.blurConversion(squares32,5,0)
squares64 = fun.blurConversion(squares64,5,0)
squares128 = fun.blurConversion(squares128,5,0)

circles16 = fun.blurConversion(circles16, 5, 0)
circles32 = fun.blurConversion(circles32,5 ,0)
circles64 = fun.blurConversion(circles64,5,0)
circles128 = fun.blurConversion(circles128,5,0)

triangles16 = fun.blurConversion(triangles16,5,0)
triangles32 = fun.blurConversion(triangles32,5,0)
triangles64 = fun.blurConversion(triangles64,5,0)
triangles128 = fun.blurConversion(triangles128,5,0)

trapezia16 = fun.blurConversion(trapezia16,5,0)
trapezia32 = fun.blurConversion(trapezia32,5,0)
trapezia64 = fun.blurConversion(trapezia64,5,0)
trapezia128 = fun.blurConversion(trapezia128,5,0)

rhombuses16 = fun.blurConversion(rhombuses16,5,0)
rhombuses32 = fun.blurConversion(rhombuses32,5,0)
rhombuses64 = fun.blurConversion(rhombuses64,5,0)
rhombuses128 = fun.blurConversion(rhombuses128,5,0)

rectangles16 = fun.blurConversion(rectangles16,5,0)
rectangles32 = fun.blurConversion(rectangles32,5,0)
rectangles64 = fun.blurConversion(rectangles64,5,0)
rectangles128 = fun.blurConversion(rectangles128,5,0)

lines16 = fun.blurConversion(lines16,5,0)
lines32 = fun.blurConversion(lines32,5,0)
lines64 = fun.blurConversion(lines64,5,0)
lines128 = fun.blurConversion(lines128,5,0)

hexagons16 = fun.blurConversion(hexagons16,5,0)
hexagons32 = fun.blurConversion(hexagons32,5,0)
hexagons64 = fun.blurConversion(hexagons64,5,0)
hexagons128 = fun.blurConversion(hexagons128,5,0)

ellipsis16 = fun.blurConversion(ellipsis16,5,0)
ellipsis32 = fun.blurConversion(ellipsis32,5,0)
ellipsis64 = fun.blurConversion(ellipsis64,5,0)
ellipsis128 = fun.blurConversion(ellipsis128,5,0)


#convertendo para binária
squares16 = fun.binaryConversion(squares16,255,31)
squares32 = fun.binaryConversion(squares32,255,31)
squares64 = fun.binaryConversion(squares64,255,31)
squares128 = fun.binaryConversion(squares128,255,31)

circles16 = fun.binaryConversion(circles16, 255, 31)
circles32 = fun.binaryConversion(circles32,255,31)
circles64 = fun.binaryConversion(circles64,255,31)
circles128 = fun.binaryConversion(circles128,255,31)

triangles16 = fun.binaryConversion(triangles16,255,31)
triangles32 = fun.binaryConversion(triangles32,255,31)
triangles64 = fun.binaryConversion(triangles64,255,31)
triangles128 = fun.binaryConversion(triangles128,255,31)

trapezia16 = fun.binaryConversion(trapezia16,255,31)
trapezia32 = fun.binaryConversion(trapezia32,255,31)
trapezia64 = fun.binaryConversion(trapezia64,255,31)
trapezia128 = fun.binaryConversion(trapezia128,255,31)

rhombuses16 = fun.binaryConversion(rhombuses16,255,31)
rhombuses32 = fun.binaryConversion(rhombuses32,255,31)
rhombuses64 = fun.binaryConversion(rhombuses64,255,31)
rhombuses128 = fun.binaryConversion(rhombuses128,255,31)

rectangles16 = fun.binaryConversion(rectangles16,255,31)
rectangles32 = fun.binaryConversion(rectangles32,255,31)
rectangles64 = fun.binaryConversion(rectangles64,255,31)
rectangles128 = fun.binaryConversion(rectangles128,255,31)

lines16 = fun.binaryConversion(lines16,255,31)
lines32 = fun.binaryConversion(lines32,255,31)
lines64 = fun.binaryConversion(lines64,255,31)
lines128 = fun.binaryConversion(lines128,255,31)

hexagons16 = fun.binaryConversion(hexagons16,255,31)
hexagons32 = fun.binaryConversion(hexagons32,255,31)
hexagons64 = fun.binaryConversion(hexagons64,255,31)
hexagons128 = fun.binaryConversion(hexagons128,255,31)

ellipsis16 = fun.binaryConversion(ellipsis16,255,31)
ellipsis32 = fun.binaryConversion(ellipsis32,255,31)
ellipsis64 = fun.binaryConversion(ellipsis64,255,31)
ellipsis128 = fun.binaryConversion(ellipsis128,255,31)

#invertendo as cores

squares16 = fun.invertConversion(squares16)
squares32 = fun.invertConversion(squares32)
squares64 = fun.invertConversion(squares64)
squares128 = fun.invertConversion(squares128)

circles16 = fun.invertConversion(circles16)
circles32 = fun.invertConversion(circles32)
circles64 = fun.invertConversion(circles64)
circles128 = fun.invertConversion(circles128)

triangles16 = fun.invertConversion(triangles16)
triangles32 = fun.invertConversion(triangles32)
triangles64 = fun.invertConversion(triangles64)
triangles128 = fun.invertConversion(triangles128)

trapezia16 = fun.invertConversion(trapezia16)
trapezia32 = fun.invertConversion(trapezia32)
trapezia64 = fun.invertConversion(trapezia64)
trapezia128 = fun.invertConversion(trapezia128)

rhombuses16 = fun.invertConversion(rhombuses16)
rhombuses32 = fun.invertConversion(rhombuses32)
rhombuses64 = fun.invertConversion(rhombuses64)
rhombuses128 = fun.invertConversion(rhombuses128)

rectangles16 = fun.invertConversion(rectangles16)
rectangles32 = fun.invertConversion(rectangles32)
rectangles64 = fun.invertConversion(rectangles64)
rectangles128 = fun.invertConversion(rectangles128)

lines16 = fun.invertConversion(lines16)
lines32 = fun.invertConversion(lines32)
lines64 = fun.invertConversion(lines64)
lines128 = fun.invertConversion(lines128)

hexagons16 = fun.invertConversion(hexagons16)
hexagons32 = fun.invertConversion(hexagons32)
hexagons64 = fun.invertConversion(hexagons64)
hexagons128 = fun.invertConversion(hexagons128)

ellipsis16 = fun.invertConversion(ellipsis16)
ellipsis32 = fun.invertConversion(ellipsis32)
ellipsis64 = fun.invertConversion(ellipsis64)
ellipsis128 = fun.invertConversion(ellipsis128)
print('terminou pre processing')

# extraindo caracteristicas das imagens
'''



squares128_vector = fun.extratorCaracteristicas(squares128)
circles128_vector = fun.extratorCaracteristicas(circles128)
triangles128_vector = fun.extratorCaracteristicas(triangles128)
trapezia128_vector = fun.extratorCaracteristicas(trapezia128)
rhombuses128_vector = fun.extratorCaracteristicas(rhombuses128)
rectangles128_vector = fun.extratorCaracteristicas(rectangles128)
lines128_vector = fun.extratorCaracteristicas(lines128)
hexagons128_vector = fun.extratorCaracteristicas(hexagons128)
ellipsis128_vector = fun.extratorCaracteristicas(ellipsis128)

squares64_vector = fun.extratorCaracteristicas(squares64)
circles64_vector = fun.extratorCaracteristicas(circles64)
triangles64_vector = fun.extratorCaracteristicas(triangles64)
trapezia64_vector = fun.extratorCaracteristicas(trapezia64)
rhombuses64_vector = fun.extratorCaracteristicas(rhombuses64)
rectangles64_vector = fun.extratorCaracteristicas(rectangles64)
lines64_vector = fun.extratorCaracteristicas(lines64)
hexagons64_vector = fun.extratorCaracteristicas(hexagons64)
ellipsis64_vector = fun.extratorCaracteristicas(ellipsis64)

squares32_vector = fun.extratorCaracteristicas(squares32)
circles32_vector = fun.extratorCaracteristicas(circles32)
triangles32_vector = fun.extratorCaracteristicas(triangles32)
trapezia32_vector = fun.extratorCaracteristicas(trapezia32)
rhombuses32_vector = fun.extratorCaracteristicas(rhombuses32)
rectangles32_vector = fun.extratorCaracteristicas(rectangles32)
lines32_vector = fun.extratorCaracteristicas(lines32)
hexagons32_vector = fun.extratorCaracteristicas(hexagons32)
ellipsis32_vector = fun.extratorCaracteristicas(ellipsis32)
'''

squares16_vector = fun.extratorCaracteristicas(squares16)
circles16_vector = fun.extratorCaracteristicas(circles16)
triangles16_vector = fun.extratorCaracteristicas(triangles16)
trapezia16_vector = fun.extratorCaracteristicas(trapezia16)
rhombuses16_vector = fun.extratorCaracteristicas(rhombuses16)
rectangles16_vector = fun.extratorCaracteristicas(rectangles16)
lines16_vector = fun.extratorCaracteristicas(lines16)
hexagons16_vector = fun.extratorCaracteristicas(hexagons16)
ellipsis16_vector = fun.extratorCaracteristicas(ellipsis16)


print('terminou extracao carac')

# transformando os vetores em dataframes
'''

squares128_vector = pd.DataFrame(squares128_vector)
circles128_vector = pd.DataFrame(circles128_vector)
triangles128_vector = pd.DataFrame(triangles128_vector)
trapezia128_vector = pd.DataFrame(trapezia128_vector)
rhombuses128_vector = pd.DataFrame(rhombuses128_vector)
rectangles128_vector = pd.DataFrame(rectangles128_vector)
lines128_vector = pd.DataFrame(lines128_vector)
hexagons128_vector = pd.DataFrame(hexagons128_vector)
ellipsis128_vector = pd.DataFrame(ellipsis128_vector)

squares32_vector = pd.DataFrame(squares32_vector)
circles32_vector = pd.DataFrame(circles32_vector)
triangles32_vector = pd.DataFrame(triangles32_vector)
trapezia32_vector = pd.DataFrame(trapezia32_vector)
rhombuses32_vector = pd.DataFrame(rhombuses32_vector)
rectangles32_vector = pd.DataFrame(rectangles32_vector)
hexagons32_vector = pd.DataFrame(hexagons32_vector)
ellipsis32_vector = pd.DataFrame(ellipsis32_vector)
lines32_vector = pd.DataFrame(lines32_vector)

squares64_vector = pd.DataFrame(squares64_vector)
circles64_vector = pd.DataFrame(circles64_vector)
triangles64_vector = pd.DataFrame(triangles64_vector)
trapezia64_vector = pd.DataFrame(trapezia64_vector)
rhombuses64_vector = pd.DataFrame(rhombuses64_vector)
rectangles64_vector = pd.DataFrame(rectangles64_vector)
lines64_vector = pd.DataFrame(lines64_vector)
hexagons64_vector = pd.DataFrame(hexagons64_vector)
ellipsis64_vector = pd.DataFrame(ellipsis64_vector)
'''

circles16_vector = pd.DataFrame(circles16_vector)
squares16_vector = pd.DataFrame(squares16_vector)
triangles16_vector = pd.DataFrame(triangles16_vector)
trapezia16_vector = pd.DataFrame(trapezia16_vector)
rhombuses16_vector = pd.DataFrame(rhombuses16_vector)
rectangles16_vector = pd.DataFrame(rectangles16_vector)
lines16_vector = pd.DataFrame(lines16_vector)
hexagons16_vector = pd.DataFrame(hexagons16_vector)
ellipsis16_vector = pd.DataFrame(ellipsis16_vector)


print('terminou transformar em dataframe')

#incluindo a classe nos dataframes
'''
squares128_vector['Classe'] = 'square'
circles128_vector['Classe'] = 'circle'
triangles128_vector['Classe'] = 'triangle'
trapezia128_vector['Classe'] = 'trapezia'
rhombuses128_vector['Classe'] = 'rhombuse'
rectangles128_vector['Classe'] = 'rectangle'
lines128_vector['Classe'] = 'line'
hexagons128_vector['Classe'] = 'hexagon'
ellipsis128_vector['Classe'] = 'ellipse'

squares32_vector['Classe'] = 'square'
circles32_vector['Classe'] = 'circle'
triangles32_vector['Classe'] = 'triangle'
trapezia32_vector['Classe'] = 'trapezia'
rhombuses32_vector['Classe'] = 'rhombuse'
rectangles32_vector['Classe'] = 'rectangle'
lines32_vector['Classe'] = 'line'
hexagons32_vector['Classe'] = 'hexagon'
ellipsis32_vector['Classe'] = 'ellipse'

squares64_vector['Classe'] = 'square'
circles64_vector['Classe'] = 'circle'
triangles64_vector['Classe'] = 'triangle'
trapezia64_vector['Classe'] = 'trapezia'
rhombuses64_vector['Classe'] = 'rhombuse'
rectangles64_vector['Classe'] = 'rectangle'
lines64_vector['Classe'] = 'line'
hexagons64_vector['Classe'] = 'hexagon'
ellipsis64_vector['Classe'] = 'ellipse'

'''

squares16_vector['Classe'] = 'square'
circles16_vector['Classe'] = 'circle'
triangles16_vector['Classe'] = 'triangle'
trapezia16_vector['Classe'] = 'trapezia'
rhombuses16_vector['Classe'] = 'rhombuse'
rectangles16_vector['Classe'] = 'rectangle'
lines16_vector['Classe'] = 'line'
hexagons16_vector['Classe'] = 'hexagon'
ellipsis16_vector['Classe'] = 'ellipse'

'''

dfs64 = [squares64_vector,circles64_vector,triangles64_vector,trapezia64_vector,rhombuses64_vector,
         rectangles64_vector,lines64_vector,hexagons64_vector,ellipsis64_vector]


dfs128 = [squares128_vector,circles128_vector,triangles128_vector,trapezia128_vector,rhombuses128_vector,
          rectangles128_vector,lines128_vector,hexagons128_vector,ellipsis128_vector]

dfs32 = [squares32_vector,circles32_vector,triangles32_vector,trapezia32_vector,rhombuses32_vector,
         rectangles32_vector,lines32_vector,hexagons32_vector,ellipsis32_vector]
'''

dfs16 = [squares16_vector,circles16_vector,triangles16_vector,trapezia16_vector,rhombuses16_vector,
       rectangles16_vector,lines16_vector,hexagons16_vector,ellipsis16_vector]



# USANDO AS IMAGENS 128x128
'''
dataFrame128 = pd.concat(dfs128, ignore_index=True)
dataFrame128.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Images_128.csv')
dataFrame128_2 = dataFrame128.copy()
del dataFrame128['Classe']
dataFrame128 = fun.normalizar(dataFrame128)
dataFrame128.fillna(0)
dataFrame128['Classe'] = dataFrame128_2['Classe']
'''



# Criando o objeto do tipo k-folds com 10 folds
kfold = KFold(10, True, 1)

# Instanciando os algoritmos e seus vetores de tempo e acurácia

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
'''
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
'''

## USANDO A BASE COM IMAGENS DE 64x64
'''
dataFrame64 = pd.concat(dfs64, ignore_index=True)
print(dataFrame64)
dataFrame64.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Images_64.csv')
dataFrame64 = dataFrame64.fillna(0)
print(dataFrame64)
dataFrame64_2 = dataFrame64.copy()
del dataFrame64['Classe']

dataFrame64 = fun.normalizar(dataFrame64)

dataFrame64['Classe'] = dataFrame64_2['Classe']

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

dataFrame32 = pd.concat(dfs32, ignore_index=True)
dataFrame32.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Images_32.csv')
dataFrame32 = dataFrame32.fillna(0)

dataFrame32_2 = dataFrame32.copy()
del dataFrame32['Classe']

dataFrame32 = fun.normalizar(dataFrame32)

dataFrame32['Classe'] = dataFrame32_2['Classe']

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
# USANDO A BASE COM IMAGENS 16x16

dataFrame16 = pd.concat(dfs16, ignore_index=True)
dataFrame16.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Images_16.csv')
dataFrame16 = dataFrame16.fillna(0)
dataFrame16_2 = dataFrame16.copy()
del dataFrame16['Classe']
dataFrame16 = fun.normalizar(dataFrame16)
dataFrame16['Classe'] = dataFrame16_2['Classe']

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
