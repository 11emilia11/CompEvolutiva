import pandas as pd
from numpy import ndarray
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from sklearn.naive_bayes import GaussianNB
import statistics
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
import glob
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_similarity_score
from operator import itemgetter
import math


# função que calcula a quantidade de elementos por classe
# a partir do data frame original, o nome da classe e a classificação

def quantidade_por_classe (dados, nome_classe, classe):
    cont = 0
    for x in range(len(dados.index)):
        if (dados[nome_classe].iloc[x] == classe):
            cont += 1
    return cont
# função de inicialização do algoritmo KNN , recebe o k
# e a distância que vai ser usada como referência

def inicializacao_KNN (k):

    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    return knn


# função de normalização dos dados usando o modelo de
# normalização por reescala

def normalizar(dados):
    x = dados.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    dados_norm = pd.DataFrame(x_scaled)
    return dados_norm

# função que calcula os valores de média, moda , mediana e desvio padrão
# para os vetores de acerto de um algoritmo, recebe como parâmetro o vetor
# de acerto e o nome do algoritmo

def tendencia_central (nomeAlgoritmo, vetorAcerto, vetorTempo):
    print('________________________________________________\n')
    print(nomeAlgoritmo)
    print('Tempo Média = ', np.mean(vetorTempo))
    print('Tempo Desvio padrão = ', statistics.pstdev(vetorTempo))
    print('Tempo Moda = ', stats.mode(vetorTempo, axis=None))
    print('Tempo Mediana =', np.median(vetorTempo))
    print('----------------------------------------------')
    print('Acurácia Média = ', np.mean(vetorAcerto))
    print('Acurácia Desvio padrão = ', statistics.pstdev(vetorAcerto))
    print('Acurácia Moda = ', stats.mode(vetorAcerto, axis=None))
    print('Acurácia Mediana = ', np.median(vetorAcerto))

    print('________________________________________________\n')

# função que cria amostras estratificadas a partir
# dos Data frame, o tamanho desejado para a amostra
# e a classe dos dados

def amostra_estrat(dados, tamanho_amostra, classe):
    classes = dados[classe].unique()
    qtde_por_classe = round(tamanho_amostra / len(classes))
    amostras_por_classe = []
    for c in classes:
        indices_c = dados[classe] == c
        obs_c = dados[indices_c]
        amostra_c = obs_c.sample(qtde_por_classe)
        amostras_por_classe.append(amostra_c)

    amostra_estratificada = pd.concat(amostras_por_classe)
    return amostra_estratificada

# função que realiza o treinamento dos algoritmos usados na base de dados

def treinaralgoritmos(noclass_train, class_train , tree, knnp1 , knnp2 , knnp3, knn1 , knn2 , knn3, naive, svmlinear , svmrbf):

    knn1.fit(noclass_train, class_train)
    knn2.fit(noclass_train, class_train)
    knn3.fit(noclass_train, class_train)
    naive.fit(noclass_train, class_train)
    tree.fit(noclass_train, class_train)
    knnp1.fit(noclass_train, class_train)
    knnp2.fit(noclass_train, class_train)
    knnp3.fit(noclass_train, class_train)
    svmlinear.fit(noclass_train, class_train)
    svmrbf.fit(noclass_train, class_train)

# função de inicialização do algoritmo KNN Ponderado
# recebe como parâmentro o valor do k

def inicializando_KNNW (k):

    knnp = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
    return knnp



def geneticlabels(dataframe,centers):
    return pairwise_distances_argmin_min(dataframe,centers,metric='minkowski')

def find_clusters(X, n_clusters, rng, max_it):

    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    max_iterator = 0
    distances = []
    while True:

        labels,distance = pairwise_distances_argmin_min(X,centers,metric='minkowski')
        distances.append(distance)

        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])


        if np.all(centers == new_centers) or max_iterator > max_it:
            break
        centers = new_centers
        max_iterator += 1

    return centers, labels, distances

def accuracy_majority_vote(base, predict_labels, real_labels, n_clusters):

    classes = real_labels.unique()

    majority = []
    groups = []
    k = 0
    for i in range(n_clusters):
        group = []
        for a in range(len(base)):
            if predict_labels[a] == i:
                group.append(real_labels[a])
        groups.append(group)
        majority.append(initialize_dic_majority(classes))
        for real_label in group:
            majority[k][real_label] += 1
        k += 1

    label_groups = []
    for m in majority:
        label_groups.append(max(m.items(), key=itemgetter(1))[0])

    pred_labels = []
    true_labels = []
    for g in range(len(groups)):
        pred_labels = pred_labels + ([label_groups[g]]*len(groups[g]))
        true_labels = true_labels + [a for a in groups[g]]

    return accuracy_score(pred_labels, true_labels)


def find_clustersGENETIC(X, n_clusters, max_it, array):

    centers = array

    max_iterator = 0
    distances = []
    while True:

        labels,distance = pairwise_distances_argmin_min(X,centers,metric='minkowski')
        distances.append(distance)

        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])


        if np.all(centers == new_centers) or max_iterator > max_it:
            break
        centers = new_centers
        max_iterator += 1

    return centers, labels, distances

# Carregando fotos da pasta
def loadFiles(path, array):

    for i in glob.glob(path):

        img = cv2.imread(i)
        array.append(img)

    return array

# Função que aplic o filtro blur nas fotos do array
def blurConversion(arrayphotos ,val1, val2):

    for x in range(len(arrayphotos)):
        arrayphotos[x] = cv2.GaussianBlur(arrayphotos[x],(val1,val1), val2)

    return arrayphotos

#Função que faz a binarização das fotos
def binaryConversion(arrayphotos,threshold,val1):

    for x in range(len(arrayphotos)):
        arrayphotos[x] = cv2.adaptiveThreshold(arrayphotos[x],threshold,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,val1,10)

    return arrayphotos

#Função que inverte as fotos binárias
def invertConversion(arrayphotos):

    for x in range(len(arrayphotos)):
        arrayphotos[x] = cv2.bitwise_not(arrayphotos[x])

    return arrayphotos

# Função que faz o filtro cinza nas fotos
def grayConversion(arrayphotos):

    size = len(arrayphotos)
    for x in range (0,size):
        arrayphotos[x] = cv2.cvtColor(arrayphotos[x], cv2.COLOR_BGR2GRAY)

    return arrayphotos


def WCSSgenetic(x, population):

    arrayint = []
    for a in x:
        arrayint.append(int(a))

    print(arrayint)
    soma = 0
    for b in arrayint:
        labels, distances = pairwise_distances_argmin_min(population[b],population, metric='minkowski')
        for x in distances:
            soma += x**2
    return soma


def generatepopulation(X,numberpopu, K, rng):

    population = []

    for x in range(numberpopu):
        first = rng.permutation(X.shape[0])[:K]
        print(first)
        population.append(np.concatenate(X[first]))

    return population


def find_clustersgenetic(X, n_clusters, max_it, array):

    centers = array

    max_iterator = 0
    distances = []
    while True:

        labels,distance = pairwise_distances_argmin_min(X,centers,metric='minkowski')
        distances.append(distance)

        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])

        if np.all(centers == new_centers) or max_iterator > max_it:
            break
        centers = new_centers
        max_iterator += 1

    return centers, labels, distances


#FUNÇÃO DE NORMALIZAÇÃO


def normalize(df1):
    x = df1.values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(scaled)
    return df_normalized

def normalizearray(array):
    scaled = preprocessing.MinMaxScaler().fit_transform(array)
    return scaled

def normalizeArrayofArrays(arrayphotos):

    size = len(arrayphotos)
    for x in range (0,size):
        arrayphotos[x] = preprocessing.MinMaxScaler().fit_transform(arrayphotos[x])

    return arrayphotos

def Turntogray(arrayphotos):

    size = len(arrayphotos)
    for x in range (0,size):
        arrayphotos[x] = cv2.cvtColor(arrayphotos[x], cv2.COLOR_BGR2GRAY)

    return arrayphotos


def gaussianblurArray(arrayphotos,val1,val2):

    for x in range(len(arrayphotos)):
        arrayphotos[x] = cv2.GaussianBlur(arrayphotos[x],(val1,val1), val2)

    return arrayphotos

def binaryadaptive(arrayphotos,threshold,val1):

    for x in range(len(arrayphotos)):
        arrayphotos[x] = cv2.adaptiveThreshold(arrayphotos[x],threshold,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,val1,10)

    return arrayphotos

def initialize_dic_majority(classes):
    majority = {}
    for c in classes:
        majority[c] = 0

    return majority

def invertbinaryphotos(arrayphotos):

    for x in range(len(arrayphotos)):
        arrayphotos[x] = cv2.bitwise_not(arrayphotos[x])

    return arrayphotos


def loadfolderimgs(path):

    arrayphotos = []

    for img in glob.glob(path):
        n = cv2.imread(img)
        arrayphotos.append(n)

    return arrayphotos


def imgtoarray(arrayphotos):

    size = len(arrayphotos)
    for x in range (0,size):
        arrayphotos[x] = np.array(arrayphotos[x] , dtype=float)

    return arrayphotos

def gethumoments(arrayphotos):

    for x in range(len(arrayphotos)):

        arrayphotos[x] = cv2.HuMoments(cv2.moments(arrayphotos[x]), True).flatten()

    return arrayphotos

def WCSS2(distance):
    soma = 0
    for x in distance:
        soma += x ** 2

    return soma

def extratorCaracteristicas(arrayimgs):

    squarescarac = []


    for x in arrayimgs:

        aux = []

        im2, countours, hierachy = cv2.findContours(x, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(countours) == 0:

            for b in range(0,45):

                aux.append(0)

        elif len(countours) > 0:
            peri = cv2.arcLength(countours[0], True)  # perimetro
            aux.append(peri)

            aproxx = cv2.approxPolyDP(countours[0], 0.04 * peri, True)  # vertices
            vertc = len(aproxx)
            aux.append(vertc)

            area = cv2.contourArea(countours[0])  # area
            aux.append(area)

            if peri > 0:
                compactness = (4 * math.pi * area) / (peri ** 2)  # compacidade
                aux.append(compactness)
            elif peri == 0:
                aux.append(0)

            momentum = cv2.moments(x)
            momentum = list(dict.values(momentum))  # momentos
            for i in momentum:
                aux.append(i)

            moments = cv2.HuMoments(momentum, True).flatten()  # Hu moments

            for i in moments:
                aux.append(i)

            histogram = get_histogram(aproxx)  # Frequencia de angulos

            for h in histogram:
                aux.append(h)

        squarescarac.append(aux)


    return squarescarac

def niveisCinza(image):

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    return gray


def filtroGaussiano(image):

    gaussianb = cv2.GaussianBlur(image,(5,5),0)

    return gaussianb

def turnToBinary(image):

    binary = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,10)

    return binary

def inverterCores(image):

    invert = cv2.bitwise_not(image)

    return invert

def resizeImages(images,width,height):

    i = len(images)
    for x in range(0,i):

        images[x] = cv2.resize(images[x],(width,height))

    return images


def angle3pt(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))

    return ang + 360 if ang < 0 else ang

def get_vertices(imagem):

    im2, countours, hierachy = cv2.findContours(imagem, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    peri = cv2.arcLength(countours[0], True)  # perimetro

    aproxx = cv2.approxPolyDP(countours[0], 0.04 * peri, True)  # vertices

    return aproxx

def get_angle(p0, p1, p2):

    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return np.degrees(angle)


def get_histogram(aproxx):
    zero = []
    zero.append(0)
    zero.append(0)
    zero.append(0)
    zero.append(0)
    zero.append(0)
    zero.append(0)
    zero.append(0)
    zero.append(0)
    zero.append(0)
    zero.append(0)
    zero.append(0)
    #transformando em um array bidimensional
    retornopollyDP = np.reshape(aproxx, (aproxx.shape[0], (aproxx.shape[1] * aproxx.shape[2])))

    #checando se não é uma linha
    if (len(retornopollyDP) < 3):

        return zero

    arraysdeangulo = []

    for x in range(len(retornopollyDP)):
        # caso especial : quando a primeira posição do array for o angulo central
        if x == 0:

            bla3 = get_angle(retornopollyDP[x + 1], retornopollyDP[x], retornopollyDP[len(retornopollyDP) - 1])
            arraysdeangulo.append(math.fabs(bla3))

        # caso especial : quando a última posição do array for o ângulo central
        if x == len(retornopollyDP) - 1:

            bla4 = get_angle(retornopollyDP[x - 1], retornopollyDP[x], retornopollyDP[0])
            arraysdeangulo.append(math.fabs(bla4))

        if x > 0 and x < len(retornopollyDP) - 1:

            bla5 = get_angle(retornopollyDP[x + 1], retornopollyDP[x], retornopollyDP[x - 1])
            arraysdeangulo.append(math.fabs(bla5))

    hist, bins = np.histogram(arraysdeangulo, bins=[15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180])

    return hist

import pandas as pd
from numpy import ndarray
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from sklearn.naive_bayes import GaussianNB
import statistics
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import cv2
import glob
import math
import os
import random


'''

Funções para pré-processamento dos dados

'''
# Carregando fotos da pasta
def loadFiles(path, array):

    for i in glob.glob(path):

        img = cv2.imread(i)
        array.append(img)

    return array

# Função que faz o filtro cinza nas fotos
def grayConversion(arrayphotos):

    size = len(arrayphotos)
    for x in range (0,size):
        arrayphotos[x] = cv2.cvtColor(arrayphotos[x], cv2.COLOR_BGR2GRAY)

    return arrayphotos

# Função que aplic o filtro blur nas fotos do array
def blurConversion(arrayphotos ,val1, val2):

    for x in range(len(arrayphotos)):
        arrayphotos[x] = cv2.GaussianBlur(arrayphotos[x],(val1,val1), val2)

    return arrayphotos

#Função que faz a binarização das fotos
def binaryConversion(arrayphotos,threshold,val1):

    for x in range(len(arrayphotos)):
        arrayphotos[x] = cv2.adaptiveThreshold(arrayphotos[x],threshold,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,val1,10)

    return arrayphotos

#Função que inverte as fotos binárias
def invertConversion(arrayphotos):

    for x in range(len(arrayphotos)):
        arrayphotos[x] = cv2.bitwise_not(arrayphotos[x])

    return arrayphotos


# função de normalização dos dados usando o modelo de
# normalização por reescala

def normalizar(dados):
    x = dados.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    dados_norm = pd.DataFrame(x_scaled)
    return dados_norm



def extratorCaracteristicas(arrayimgs):

    squarescarac = []


    for x in arrayimgs:

        aux = []

        im2, countours, hierachy = cv2.findContours(x, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(countours) == 0:

            for b in range(0,45):

                aux.append(0)

        elif len(countours) > 0:

            momentum = cv2.moments(x)
            momentum = list(dict.values(momentum))  # momentos
            for i in momentum:
                aux.append(i)

            moments = cv2.HuMoments(momentum, True).flatten()  # Hu moments

            for i in moments:
                aux.append(i)

            area = cv2.contourArea(countours[0])  # area
            aux.append(area)

            peri = cv2.arcLength(countours[0], True)  # perimetro
            aux.append(peri)

            if peri > 0:
                compactness = (4 * math.pi * area) / (peri ** 2)  # compacidade
                aux.append(compactness)
            elif peri == 0:
                aux.append(0)

            aproxx = cv2.approxPolyDP(countours[0], 0.04 * peri, True)  # vertices
            vertc = len(aproxx)
            aux.append(vertc)

            histogram = get_histogram(aproxx)  # Histograma da Frequencia de angulos

            for h in histogram:
                aux.append(h)

        squarescarac.append(aux)


    return squarescarac


def resizeImages(images,width,height):

    i = len(images)
    for x in range(0,i):

        images[x] = cv2.resize(images[x],(width,height))

    return images


def get_angle(p0, p1, p2):

    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return np.degrees(angle)


def get_histogram(aproxx):
    zero = []
    zero.append(0)
    zero.append(0)
    zero.append(0)
    zero.append(0)
    zero.append(0)
    zero.append(0)
    zero.append(0)
    zero.append(0)
    zero.append(0)
    zero.append(0)
    zero.append(0)

    #transformando em um array bidimensional
    retornopollyDP = np.reshape(aproxx, (aproxx.shape[0], (aproxx.shape[1] * aproxx.shape[2])))

    #checando se não é uma linha
    if (len(retornopollyDP) < 3):

        return zero

    arraysdeangulo = []

    for x in range(len(retornopollyDP)):
        # caso especial : quando a primeira posição do array for o angulo central
        if x == 0:

            bla3 = get_angle(retornopollyDP[x + 1], retornopollyDP[x], retornopollyDP[len(retornopollyDP) - 1])
            arraysdeangulo.append(math.fabs(bla3))

        # caso especial : quando a última posição do array for o ângulo central
        if x == len(retornopollyDP) - 1:

            bla4 = get_angle(retornopollyDP[x - 1], retornopollyDP[x], retornopollyDP[0])
            arraysdeangulo.append(math.fabs(bla4))

        if x > 0 and x < len(retornopollyDP) - 1:

            bla5 = get_angle(retornopollyDP[x + 1], retornopollyDP[x], retornopollyDP[x - 1])
            arraysdeangulo.append(math.fabs(bla5))

    hist, bins = np.histogram(arraysdeangulo, bins=[0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180])

    return hist


# funcão que seleciona num quantidade de imagens aleatoriamente
# em um vetor e retorna um vetor auxiliar apenas com as imagens
# selecionadas

def seleciona_imagens (arrayImgs, num):
    tamanho = len(arrayImgs)
    selecionadas = []
    nao_selecionadas = []
    aux = []
    var = 0

    for x in range(0, num):

        if x == 0:

            var = randint(0, tamanho - 1)
            aux.append(var)
            selecionadas.append(arrayImgs[var])

        elif x > 0:

            var = randint(0, tamanho - 1)

            while np.isin(var ,aux):

                var = randint(0, tamanho - 1)

            selecionadas.append(arrayImgs[var])
            aux.append(var)

    for x in range(0, tamanho):
        if np.isin(x,aux):
            continue
        else:
            nao_selecionadas.append(arrayImgs[x])

    return selecionadas, nao_selecionadas


#função que salva as imagens na pasta dada como parâmetro
def save_images(images, path):
    cont = 0

    for i in images:
        y = str(cont)
        cv2.imwrite(path + y + '.jpg', i)
        cont += 1

    return 0


# função que retorna população de cromossomos e tem como parametros
# tamanho da população (número de cromossomos) e tamanho dos cromossomos
def create_population(size_population, size_chromossome):
    population = []

    for y in range(0, size_population):
        aux = []
        for i in range(0, size_chromossome):
            x = randint(0, 200)
            if x % 2 == 0:
                aux.append(0)
            elif x % 2 != 0:
                aux.append(1)
        population.append(aux)

    return population


# função que identifica o tamanho do cromossomo e retorna o vetor de características correspondente
def decode_chromossome(chromossome):

    images_class = []

    if chromossome[0] == 0 and chromossome[1] == 0:

        images_class = pd.read_csv('C:/Users/Auricelia/Desktop/DataSetsML/Images_16_s.csv')
        images_class = pd.DataFrame(images_class)

    elif chromossome[0] == 0 and chromossome[1] == 1:
        images_class = pd.read_csv('C:/Users/Auricelia/Desktop/DataSetsML/Images_32_s.csv')
        images_class = pd.DataFrame(images_class)

    elif chromossome[0] == 1 and chromossome[1] == 0:

        images_class = pd.read_csv('C:/Users/Auricelia/Desktop/DataSetsML/Images_64_s.csv')
        images_class = pd.DataFrame(images_class)

    elif chromossome[0] == 1 and chromossome[1] == 1:

        images_class = pd.read_csv('C:/Users/Auricelia/Desktop/DataSetsML/Images_128_s.csv')
        images_class = pd.DataFrame(images_class)

    return images_class

'''


def fitness(chromossome):

    if chromossome[0] == 0 and chromossome[1] == 0:

        #DOFITNESS

    elif chromossome[0] == 0 and chromossome[1] == 1:

        #DOFITNESS

    elif chromossome[0] == 1 and chromossome[1] == 0:

        #DOFITNESS

    elif chromossome[0] == 1 and chromossome[1] == 1:

        #DOFITNESS

'''

#fucção que retorna as posições do array que
#são usadas pelo cromossomo para a classificação
def positions_chromossome(chromossome):

    selected_pos = []

    for c in range(2,len(chromossome)):
        aux = []
        if chromossome[c] == 1:
            selected_pos.append(c)

        elif chromossome[c] == 0:
            continue

    return selected_pos



# função que retorna o array com características selecionadas
# pelo cromossomo passando as posicoes selecionadas
# e o vetor com as características
def carac_imagens(selected_position,array_images):
    arrayzao = []
    z = 0
    for j in range(0,len(array_images)):
        aux = []
        for i in range(0,len(selected_position)):
            z = selected_position[i]
            aux.append(array_images[j][z])

        arrayzao.append(aux)

    return arrayzao


#gera o conjunto dos dois melhores cromossomos para
#garantir o elitismo no algoritmo genético
def get_best_cromossomos(acuracias,cromossomos):
    index = []
    melhor = []
    best_acuracia = []

    print('index antes',index)
    index.append(acuracias.index(max(acuracias)))
    print('index melhor',acuracias.index(max(acuracias)))
    best_acuracia.append(max(acuracias))
    print('best acc',max(acuracias))
    acuracias[acuracias.index(max(acuracias))] = 0


    index.append(acuracias.index(max(acuracias)))
    print('index melhor',acuracias.index(max(acuracias)))
    best_acuracia.append(max(acuracias))
    print('best acc',max(acuracias))
    acuracias[acuracias.index(max(acuracias))] = 0

    print('index dpeois',index)

    for i in range(0,len(index)):
        melhor.append(cromossomos[index[i]])
        cromossomos[index[i]] = 0
        print('melhor',cromossomos[i])
    # melhor.append(cromossomos[index[0]])
    # melhor.append(cromossomos[index[1]])

    return melhor, best_acuracia, cromossomos


# seleção por torneio gera aleatoriamente dois
# cromossomos e retorna o melhor deles
def tournament_selection(acuracias, cromossomos):
    melhor = []
    aux =[]
    rand1 = randint(0,len(cromossomos)-1)
    rand2 = randint(0,len(cromossomos)-1)

    while rand2 == rand1:
        rand2 = randint(0,len(cromossomos)-1)

    aux.append(rand1)
    aux.append(rand2)


    if acuracias[rand1] > acuracias[rand2]:
        melhor.append(cromossomos[rand1])
        cromossomos[rand1] = 0

    elif acuracias[rand2] > acuracias[rand1]:
        melhor.append(cromossomos[rand2])
        cromossomos[rand2] = 0


    return melhor, cromossomos


#função que retorna o casal gerado a partir do torneio
def generate_parents(melhores):
    duplas = []
    usados = []

    for i in range(int(len(melhores)/2)):
        aux = []
        rand = randint(0, len(melhores)-1)
        if i == 0:

            aux.append(melhores[rand])
            usados.append(rand)
            rand2 = randint(0,len(melhores)-1)

            while usados.__contains__(rand2):
                rand2 = randint(0,len(melhores)-1)
            aux.append(melhores[rand2])

            usados.append(rand2)
            duplas.append(aux)


        elif i > 0:

            while usados.__contains__(rand):
                rand = randint(0,len(melhores)-1)

            aux.append(melhores[rand])
            usados.append(rand)

            rand2 = randint(0,len(melhores)-1)
            while usados.__contains__(rand2):
                rand2 = randint(0,len(melhores)-1)
            aux.append(melhores[rand2])

            usados.append(rand2)
            duplas.append(aux)


    return duplas


#função que realiza o crossover entre os dois cromossomos
#pais para a geração
def crossover(taxa_crossover, parents):

    offspring = []
    seed = random.uniform(0,1)


    if seed < taxa_crossover:
        cromo1_a = parents[0][:19]
        cromo1_b = parents[0][19:]
        cromo2_a = parents[1][:19]
        cromo2_b = parents[1][19:]
        offspring.append(cromo1_a+cromo2_b)
        offspring.append(cromo2_a+cromo1_b)

    elif seed > taxa_crossover or seed == taxa_crossover :
        offspring.append(parents[0].copy())
        offspring.append(parents[1].copy())

    return offspring


# função que inverte os bits de uma posição
# do cromossomo caso a seed seja superior à taxa de mutação
def mutation(taxa_mutation, offsp_1):
    seed = 0
    for i in range(0,len(offsp_1)):
        seed = random.uniform(0,1)
        if seed < taxa_mutation:

            if offsp_1[i] == 0:
                offsp_1[i] = 1

            elif offsp_1[i] == 1:
                offsp_1[i] = 0

        elif seed > taxa_mutation or seed == taxa_mutation:
            continue

    return offsp_1,

# função que calcula os valores de média, moda , mediana e desvio padrão
# para os vetores de acerto de um algoritmo, recebe como parâmetro o vetor
# de acerto e o nome do algoritmo

# def tendencia_central (nomeAlgoritmo, vetorAcerto, vetorTempo):
#     print('________________________________________________\n')
#     print(nomeAlgoritmo)
#     print('Tempo Média = ', np.mean(vetorTempo))
#     print('Tempo Desvio padrão = ', statistics.pstdev(vetorTempo))
#     print('Tempo Moda = ', stats.mode(vetorTempo, axis=None))
#     print('Tempo Mediana =', np.median(vetorTempo))
#     print('----------------------------------------------')
#     print('Acurácia Média = ', np.mean(vetorAcerto))
#     print('Acurácia Desvio padrão = ', statistics.pstdev(vetorAcerto))
#     print('Acurácia Moda = ', stats.mode(vetorAcerto, axis=None))
#     print('Acurácia Mediana = ', np.median(vetorAcerto))
#
#     print('________________________________________________\n')

def jaccard_index(receita1, receita2):

    index = jaccard_similarity_score(receita1, receita2)

    return index


def major_jaccard_index(receita1, dataframe):

    aux = []

    for receita in dataframe:

        jac = jaccard_index(receita1, receita)
        if jac > 0.5 or jac == 0.5:
            aux.append(jac)

    nearest = max(aux)
    posicao = aux.index(max(aux))
    classe = dataframe.at[posicao, 'Class']

    return nearest,classe

def mean_jaccard_index(receita1, dataframe):
    aux = []

    for receita in dataframe:
        jac = jaccard_index(receita1,receita)
        aux.append(jac)

    media = np.mean(aux)
    return media

# def get_probabilities():
#
#     for i in range(df_1.shape[0]):
#         soma = df_1[i].sum()
#         prob_ingrediente = soma / total_rec_1
#         p_ingrediente_1.append(prob_ingrediente)


def p_ingredientes(cromossomo, p_ingrediente_classe):

    aux = 1
    for i in range(len(cromossomo)):

        if cromossomo[i] == 1:
            aux *= p_ingrediente_classe[i]

        elif cromossomo[i] == 0:
            continue
    return aux

# função que decodifica os dois primeiros bits do cromossomo
# para identificar a sua classe
def decode_cromoossome_recipe(cromossomo):
    classe = -1
    if cromossomo[0] == 0 and cromossomo[1] == 0:
        classe = 1
    elif cromossomo[0] == 0 and cromossomo[1] == 1:
        classe = 2
    elif cromossomo[0] == 1 and cromossomo[1] == 0:
        classe = 3
    elif cromossomo[0] == 1 and cromossomo[1] == 1:
        classe = 4

    return classe




def total_ing_classe(p_ingrediente_classe):

    cont = 0

    for i in range(len(p_ingrediente_classe)):

        if p_ingrediente_classe[i] > 0:
            cont += 1

        elif p_ingrediente_classe == 0:
            continue

    return cont

def total_ing_cromossomo(p_ingrediente):

    cont = 0

    for i in range(len(p_ingrediente)):

        if p_ingrediente[i] > 0:
            cont += 1

        elif p_ingrediente == 0:
        elif p_ingrediente == 0:
            continue

    return cont


# def fitness_receita(p_ingredientes, p_ingrediente_classe):
#     fitness = p_ingredientes *
