import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import FuncoesML as fun
from scipy import stats
import numpy as np
from random import randint
from sklearn.cluster import KMeans
import pso as pso
import time
from scipy.stats import friedmanchisquare
import matplotlib.pyplot as plt

#Criando a classe receita que irá conter nome, classe e vetor de ingredientes
class Receita:
    Name = None
    Class = 0
    ingredientes = []
    ingredientesnorm = []


#Método que retorna o vetor de ingredientes
    def getingrednorm(self):
        return self.ingredientesnorm

#Construtor da classe receita
    def __init__(self, name, Class, ingredientes):
        self.Name = name
        self.Class = Class
        self.ingredientes = ingredientes

#Método que adiciona ingredientes no vetor de ingredientes
    def adicionaringrediente(self, ingrediente):
        self.ingredientes.append(ingrediente)

#abrindo o arquivo com a base de dados
reshipe = open("C:/Users/Auricelia/Desktop/DatasetsML/ReshibaseQ.txt", "rt", encoding="utf8")

#criando o vetor de receitas
receitas = []

# preenchendo o vetor de receitas
for receita in reshipe:
    dividido = receita.split(sep=',')
    dividido[(len(dividido) - 1)] = dividido[(len(dividido) - 1)].replace('\n', '')
    ingredientes = []

    for x in range(2, len(dividido)):
        ingredientes.append(dividido[x])

    receitas.append(Receita(dividido[1], dividido[0], ingredientes))

#vetor que irá receber todos os ingredientes sem repetição para fazer os vetores binários
todosingredientes = []

#preenchendo o vetor 'todosingredientes' sem repetição
for rec in receitas:
    for ingrediente in rec.ingredientes:

        if todosingredientes.__contains__(ingrediente) == False:
            todosingredientes.append(ingrediente)
#ordenando o vetor
todosingredientes = sorted(todosingredientes)

# preenchendo nos objetos receita o vetor binário com 0

for rec in receitas:
    norm = []
    for y in range(0, len(todosingredientes)):
        norm.append(0)
    rec.ingredientesnorm = norm

# Colocando 1 na posição em que existe o ingrediente

for rec in receitas:
    for y in rec.ingredientes:
        pos = todosingredientes.index(y)
        rec.ingredientesnorm[pos] = 1

# Vetor que irá receber os vetores binários de ingreientes de cada receita
arrayingredientesnorm = []

# Preenchendo o vetor com os ingredientes normalizados

for rec in receitas:
    arrayingredientesnorm.append(rec.ingredientesnorm)

# Vetor que irá receber as classes de cada receita
arrayclasse = []

# preenchendo o vetor com as classes de cada receita
for rec in receitas:
    arrayclasse.append(rec.Class)

# criando o dataframe que irá armazenar os ingredientes
df = pd.DataFrame(arrayingredientesnorm)

#adicionando a classe ao dataframe
df['Class'] = arrayclasse
lista = df['Class']
del df['Class']
df_final = np.array(df).astype(float)

#Salvando a base de dados normalizada
df.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/Receitas_norm.csv')

print('TAMANHO DF :',df_final[0].size*4)

'''
 COMEÇAM OS ALGORITMOS DE AGRUPAMENTO 

'''
print('COMEÇAM OS ALGORITMOS DE AGRUPAMENTO')
print('------------------------------------------')

KCLUSTERS = 4

def WCSS(distance):
    retorno = []
    for x in distance:
        soma = 0
        for z in x:
            soma += z ** 2
        retorno.append(soma)

    return retorno

def break_vectors(vector):
    global KCLUSTERS
    retorno = np.split(vector, KCLUSTERS)
    return retorno

def wcssGenetic(centers):
    global KCLUSTERS
    array = break_vectors(centers)

    kmeans = KMeans(KCLUSTERS, init=pd.DataFrame(array), max_iter=1, n_init=1)
    kmeans.fit(pd.DataFrame(df_final))

    return kmeans.inertia_

def generatepopulation(X, numberpopu, K, rng):
    population = []

    for x in range(numberpopu):
        first = rng.permutation(X.shape[0])[:K]
        population.append(np.concatenate(X[first]))

    return population
print(".....Começa a execução.....")
print()


# Vetores de acurácia dos algoritmos K-means e K-means híbrido
KmeansAcc = []
KmeansHbAcc = []
GeneticAcc = []

#Vetores que irão armazenar o WCSS do K-means e do K-means híbrido
WCSSkmeans = []
WCSShb = []
WCSSPSO = []

# Vetores que irão armazenar o tempo de execução de cada algoritmo
KmeansTime = []
KmeasnHBTime = []
PSOTime = []
for y in range(50):

    print("RODADA " + str(y))
    resultparcial = []
    distance = []
    h = randint(1, 10)
    rng = np.random.RandomState(h)


    print("-----------------------------------------------")

    # laço que irá executar 20 vezes o Kmeans
    auxiliar = []
    for var in range(20):

        print("VEZ KMEANS " + str(var))
        KmeansStart = time.time()
        centers, labels, distance = fun.find_clusters(df_final, KCLUSTERS, rng, 100)
        KmeansEnd = time.time()
        KmeansTime.append(KmeansEnd - KmeansStart)

        print(fun.accuracy_majority_vote(df_final, labels, lista, 2))
        auxiliar.append(fun.accuracy_majority_vote(df_final, labels, lista, 2))
        retorno = WCSS(distance)
        resultparcial.append(retorno[len(retorno) - 1])


    resultparcial = np.sort(resultparcial)
    KmeansAcc.append(np.mean(auxiliar))

    WCSSkmeans.append(resultparcial[0])

    population = generatepopulation(df_final, 20, KCLUSTERS, rng)

    PSOStart = time.time()
    p = pso(20, wcssGenetic, 0, 200000, 1608, 100, init=population)
    array = np.array(p.get_Gbest())
    array = np.split(array, 4)
    labelsgenetic, distancegenetic = fun.geneticlabels(df_final, array)
    PSOEnd = time.time()
    PSOTime.append(PSOEnd - PSOStart)

    print(fun.accuracy_majority_vote(df_final, labelsgenetic, lista, 4))
    GeneticAcc.append(fun.accuracy_majority_vote(df_final, labelsgenetic, lista, 4))
    WcssUpDate = fun.WCSS2(distancegenetic)
    WCSSPSO.append(WcssUpDate)

    print("Hybrid:")
    HybStart = time.time()

    cen, lbl, dis = fun.find_clustersgenetic(df_final, KCLUSTERS, 100, array)

    HybEnd = time.time()

    KmeasnHBTime.append(HybEnd - HybStart)

    print(fun.accuracy_majority_vote(df_final, lbl, lista, 4))
    KmeansHbAcc.append(fun.accuracy_majority_vote(df_final, lbl, lista, 4))

    ret = WCSS(dis)
    WCSShb.append(ret[len(ret) - 1])


    # num = retorno[len(retorno) - 1]
    # dictionary[h] = num

#
# print("Distancias" ,distance)

print("__________________WCSS___________________")
print()
print(np.mean(WCSSkmeans), 'Acurácia Média Kmeans')
print(np.std(WCSSkmeans), 'Desvio Padrao Kmeans')
print(np.median(WCSSkmeans),'Mediana Kmeans')
print(np.mean(WCSShb), 'Acurácia Média K hibrido')
print(np.std(WCSShb), 'Desvio padrao K hibrido')
print(np.median(WCSShb), 'Mediana K-hibrido')
print("__________________________________________")

print("__________________VOTO MAJORITARIO_______________")
print()
print('----------------Kmeans----------')
print("Kmeans media:" ,np.mean(KmeansAcc))
print('Mediana acurácia', np.median(KmeansAcc))
print('Desvio padrão', np.std(KmeansAcc))
print('----------Kmeans Hibrido-------')
print("KmeansHybrid media:" ,np.mean(KmeansHbAcc))
print('Mediana acurácia', np.median(KmeansHbAcc))
print('Desvio Padrao', np.std(KmeansHbAcc))
print('-------------PSO-------------')
print("Genetic media:" ,np.mean(GeneticAcc))
print('Mediana acurácia', np.median(GeneticAcc))
print('Desvio padrao', np.std(GeneticAcc))

print('________________Tempo de execução _______________________')
print('------------Kmeans------------------')
print('Tempo medio ', np.mean(KmeansTime))
print('Mediana tempo ', np.median(KmeansTime))
print('Desvio Padrao ', np.std(KmeansTime))


print('-----------K means hibrido -------------')

print('Tempo medio ', np.mean(KmeasnHBTime))
print('Mediana tempo ', np.median(KmeasnHBTime))
print('Tempo medio ', np.std(KmeasnHBTime))

print('----------- PSO -------------')
print('Tempo medio ', np.mean(PSOTime))
print('Mediana tempo ', np.median(PSOTime))
print('Desvio Padrao ', np.std(PSOTime))

print('FIM AGRUPAMENTO')
print('------------------------------------------')

resultadaoacuracia = []
for it in range(len(KmeansHbAcc)):
    aux = []
    aux.append(KmeansAcc[it])
    aux.append(KmeansHbAcc[it])
    aux.append(GeneticAcc[it])
    resultadaoacuracia.append(aux)

resultadaowcss = []
for it2 in range(len(WCSSPSO)):
    aux2 = []
    aux2.append(WCSSkmeans[it2])
    aux2.append(WCSShb[it2])
    aux2.append(WCSSPSO[it2])
    resultadaowcss.append(aux2)


resultadaoacuracia = pd.DataFrame(resultadaoacuracia, columns=['K-means', 'Hybrid', 'PSO'])
resultadaowcss = pd.DataFrame(resultadaowcss, columns=['K-means', 'Hybrid', 'PSO'])



stat, p = friedmanchisquare(KmeansAcc, KmeansHbAcc, GeneticAcc)
resultadaoacuracia.boxplot(figsize=(10, 6))
plt.title("P-value:" + str(p))
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.savefig("C:/Users/Auricelia/Desktop/ResultadosReshipe/AgrupamentoAcuracy.png")


stat, p = friedmanchisquare(WCSSkmeans, WCSShb, WCSSPSO)
resultadaowcss.boxplot(figsize=(10, 6))
plt.title("P-value:" + str(p))
plt.xlabel("Algorithms")
plt.ylabel("WCSS")
plt.savefig("C:/Users/Auricelia/Desktop/ResultadosReshipe/AgrupamentoWCSS.png")

'''

FIM DOS ALGORITMOS DE AGRUPMENTO 

'''
#COMEÇAM OS ALGORITMOS DE CLASSIFICAÇÃO
'''
# criando o dataframe que irá armazenar os ingredientes
df = pd.DataFrame(arrayingredientesnorm)


#adicionando a classe ao dataframe
df['Class'] = arrayclasse

print(df)

#print(df)
'''
print('Entrada',fun.quantidade_por_classe(reshipe, 'Class', 1))
print('Prato principal',fun.quantidade_por_classe(reshipe, 'Class', 2))
print('Acompanhamento',fun.quantidade_por_classe(reshipe, 'Class', 3))
print('Sobremesa',fun.quantidade_por_classe(reshipe, 'Class', 4))
'''


#df.to_csv('C:/Users/Auricelia/Desktop/DataSetsML/df_norm.csv')

#instanciando o kfold com k = 10
kfold = KFold(10, True, 1)

#instanciando os aloritmos usados

#KNN K = 3, K = 5, K = 7

K_3 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
acertoK_3 = [] # vetor que irá conter as acuráricas do algoritmo em cada um dos testes
k_3time = [] # vetor que irá conter os tempos de duração de cada algoritmo em cada um dos testes

K_5 = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
acertoK_5 = []
k_5time = []

K_7 = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
acertoK_7 = []
k_7time = []

# KNN Ponderado K = 3, K = 5, K = 7

KP_3 = KNeighborsClassifier(n_neighbors=3, weights='distance',metric='euclidean')
acertoKP_3 = []
kp3time = []

KP_5 = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
acertoKP_5 = []
kp5time = []

KP_7 = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')
acertoKP_7 = []
kp7time = []

#Naive Bayes

naiveBayes = GaussianNB()
acertonaiveBayes = []
naiveBayestime = []

#Árvore de decisão

ArvoreDecisao = DecisionTreeClassifier()
acertoArvoreDecisao= []
arvoreDecisaotime = []

#SVM linear

SVMlinear = SVC(kernel='linear')
acertoSVMLinear = []
svmlineartime = []


#SVM RBF

SVMrbf = SVC(kernel='rbf', gamma='scale')
acertoSVMrbf= []
svmrbftime = []

#Regressão Logística

logisticRegr = LogisticRegression()
logisticRarray = []
logistictime = []

# Multilayer Perceptron
MLP = MLPClassifier()
MLP_acerto = []
MLP_tempo = []

# variável que irá servir para calcular o tempo total de execução dos algoritmos
tempoinicial = time.time()


for x in range(0, 5):

    tempo1 = time.time()
    cols = list(df.columns)
    cols.remove('Class')

    # separando os dataframes um com classe outro sem classe
    df_noclass = df[cols]
    df_class = df['Class']

    # início do kfold
    c = kfold.split(df)

    for train_index, test_index in c:

        noclass_train, noclass_test = df_noclass.iloc[train_index], df_noclass.iloc[test_index]
        class_train, class_test = df_class.iloc[train_index], df_class.iloc[test_index]

        K_3start = time.time()
        K_3.fit(noclass_train, class_train)
        acertoK_3.append(K_3.score(noclass_test, class_test))
        K_3end = time.time()
        k_3time.append(K_3end - K_3start)

        K_5start = time.time()
        K_5.fit(noclass_train, class_train)
        acertoK_5.append(K_5.score(noclass_test, class_test))
        K_5end = time.time()
        k_5time.append(K_5end - K_5start)

        K_7start = time.time()
        K_7.fit(noclass_train, class_train)
        acertoK_7.append(K_7.score(noclass_test, class_test))
        K_7end = time.time()
        k_7time.append(K_7end - K_7start)

        naivestart = time.time()
        naiveBayes.fit(noclass_train, class_train)
        acertonaiveBayes.append(naiveBayes.score(noclass_test, class_test))
        naiveend = time.time()
        naiveBayestime.append(naiveend - naivestart)

        arvorestart = time.time()
        ArvoreDecisao.fit(noclass_train, class_train)
        acertoArvoreDecisao.append(ArvoreDecisao.score(noclass_test, class_test))
        treeend = time.time()
        arvoreDecisaotime.append(treeend - arvorestart)

        kp3start = time.time()
        KP_3.fit(noclass_train, class_train)
        acertoKP_3.append(KP_3.score(noclass_test, class_test))
        kp3end = time.time()
        kp3time.append(kp3end - kp3start)

        kp7start = time.time()
        KP_7.fit(noclass_train, class_train)
        acertoKP_7.append(KP_7.score(noclass_test, class_test))
        kp7end = time.time()
        kp7time.append(kp7end - kp7start)

        kp5start = time.time()
        KP_5.fit(noclass_train, class_train)
        acertoKP_5.append(KP_5.score(noclass_test, class_test))
        kp5end = time.time()
        kp5time.append(kp5end - kp5start)

        svmlinearstart = time.time()
        SVMlinear.fit(noclass_train, class_train)
        acertoSVMLinear.append(SVMlinear.score(noclass_test, class_test))
        svmlinearend = time.time()
        svmlineartime.append(svmlinearend - svmlinearstart)

        svmrbfstart = time.time()
        SVMrbf.fit(noclass_train, class_train)
        acertoSVMrbf.append(SVMrbf.score(noclass_test, class_test))
        svmrbfend = time.time()
        svmrbftime.append(svmrbfend - svmrbfstart)

        logisticstart = time.time()
        logisticRegr.fit(noclass_train, class_train)
        logisticRarray.append(logisticRegr.score(noclass_test, class_test))
        logisticend = time.time()
        logistictime.append(logisticend - logisticstart)

        MLP_inicio = time.time()
        MLP.fit(noclass_train, class_train)
        MLP_acerto.append(MLP.score(noclass_test, class_test))
        MLP_fim = time.time()
        MLP_tempo.append(MLP_fim - MLP_inicio)


    df = df.sample(frac=1)
    print("Terminou a ", x)
    tempo2 = time.time()
    print("Tempo da rodada ", x, (tempo2 - tempo1) / 60)

tempofinal = time.time()

mediaknn3 = np.mean(acertoK_3)
medianaknn3 = np.median(acertoK_3)
stdknn3 = np.std(acertoK_3)
timeknn3 = np.mean(k_3time)

mediaknn5 = np.mean(acertoK_5)
medianaknn5 = np.median(acertoK_5)
stdknn5 = np.std(acertoK_5)
timeknn5 = np.mean(k_5time)

mediaknn7 = np.mean(acertoK_7)
medianaknn7 = np.median(acertoK_7)
stdknn7 = np.std(acertoK_7)
timeknn7 = np.mean(k_7time)

mediaMLP = np.mean(MLP_acerto)
medianaMLP = np.median(MLP_acerto)
stdMLP = np.std(MLP_acerto)
timeMLP = np.median(MLP_tempo)


print("_______________________________________________")
print("Multi Layer Perceptron")
print("Media: ", mediaMLP)
print("Mediana: ", medianaMLP)
print("Desvio padrão: ", stdMLP)
print("Tempo médio: ", timeMLP)
print("_______________________________________________")



print('________________________________________________\n')
print("KNN")
print("Media:\nK = 3: ", mediaknn3, " K = 5: ", mediaknn5, " K = 7: ", mediaknn7)
print("Mediana:\nK = 3: ", medianaknn3, " K = 5: ", medianaknn5, " K = 7: ", medianaknn7)
print("Desvio Padrão:\nK = 3: ", stdknn3, " K = 5: ", stdknn5, " K = 7: ", stdknn7)
print("Tempo médio:\nK = 3: ", timeknn3, " K = 5: ", timeknn5, " K = 7: ", timeknn7)
print("_______________________________________________")

mediaknnpounded3 = np.mean(acertoKP_3)
medianaknnpounded3 = np.median(acertoKP_3)
stdknnpounded3 = np.std(acertoKP_3)
timewknn3 = np.mean(kp3time)

mediaknnpounded5 = np.mean(acertoKP_5)
medianaknnpounded5 = np.median(acertoKP_5)
stdknnpounded5 = np.std(acertoKP_5)
timewknn5 = np.mean(kp5time)

mediaknnpounded7 = np.mean(acertoKP_7)
medianaknnpounded7 = np.median(acertoKP_7)
stdknnpounded7 = np.std(acertoKP_7)
timewknn7 = np.mean(kp7time)

print("_______________________________________________")
print("KNN Ponderado ")
print("Media:\nk = 3: ", mediaknnpounded3, " k = 5: ", mediaknnpounded5, " k = 7: ", mediaknnpounded7)
print("Mediana:\nk = 3: ", medianaknnpounded3, " k = 5: ", medianaknnpounded5, " k = 7: ", medianaknnpounded7)
print("Desvio padrão:\nk = 3: ", stdknnpounded3, " k = 5: ", stdknnpounded5, " k = 7: ", stdknnpounded7)
print("Tempo médio:\nk = 3: ", timewknn3, " k = 5: ", timewknn5, " k = 5: ", timewknn7)
print("_______________________________________________")

medianaive = np.mean(acertonaiveBayes)
mediananaive = np.median(acertonaiveBayes)
stdnaive = np.std(acertonaiveBayes)
timenaive = np.mean(naiveBayestime)

print("_______________________________________________")
print("Naïve Bayes")
print("Media: ", medianaive)
print("Mediana: ", mediananaive)
print("Desvio padrão: ", stdnaive)
print("Tempo médio: ", timenaive)
print("_______________________________________________")

mediatree = np.mean(acertoArvoreDecisao)
medianatree = np.median(acertoArvoreDecisao)
stdtree = np.std(acertoArvoreDecisao)
timetree = np.mean(arvoreDecisaotime)

print("_______________________________________________")
print("Árvore de decisão")
print("Media: ", mediatree)
print("Mediana: ", medianatree)
print("Desvio padrão: ", stdtree)
print("Tempo médio: ", timetree)
print("_______________________________________________")

mediasvmlinear = np.mean(acertoSVMLinear)
medianasvmlinear = np.median(acertoSVMLinear)
stdsvmlinear = np.std(acertoSVMLinear)
timesvmlinear = np.mean(svmlineartime)

print("_______________________________________________")
print("SVM kernel linear")
print("Media: ", mediasvmlinear)
print("Mediana: ", medianasvmlinear)
print("Desvio padrão: ", stdsvmlinear)
print("Tempo médio: ", timesvmlinear)
print("_______________________________________________")

mediasvmrbf = np.mean(acertoSVMrbf)
medianasvmrbf = np.median(acertoSVMrbf)
stdsvmrbf = np.std(acertoSVMrbf)
timesvmrbf = np.mean(svmrbftime)

print("_______________________________________________")
print("SVM kernel rbf")
print("Media: ", mediasvmrbf)
print("Mediana: ", medianasvmrbf)
print("Desvio padrão: ", stdsvmrbf)
print("Tempo médio: ", timesvmrbf)
print("_______________________________________________")

medialogistic = np.mean(logisticRarray)
medianalogistic = np.median(logisticRarray)
stdslogistic = np.std(logisticRarray)
timelogistic = np.mean(logistictime)

print("_______________________________________________")
print("Regressao Logistica")
print("Media: ", medialogistic)
print("Mediana: ", medianalogistic)
print("Desvio padrão: ", stdslogistic)
print("Tempo médio: ", timelogistic)
print("_______________________________________________")


print("Tempo total: ", (tempofinal - tempoinicial) / 60)


resultadocsv = []
for x in range(len(acertoKP_3)):
    aux = []
    aux.append(acertoK_3[x])
    aux.append(acertoK_5[x])
    aux.append(acertoK_7[x])
    aux.append(acertoKP_3[x])
    aux.append(acertoKP_5[x])
    aux.append(acertoKP_7[x])
    aux.append(acertonaiveBayes[x])
    aux.append(acertoArvoreDecisao[x])
    aux.append(acertoSVMLinear[x])
    aux.append(acertoSVMrbf[x])
    aux.append(logisticRarray[x])
    aux.append(MLP_acerto[x])
    resultadocsv.append(aux)


resultadocsv = pd.DataFrame(resultadocsv,columns=['KNN3','KNN5','KNN7','KNNP3','KNNP5','KNNP7',
'NAIVE','TREE','SVMLINEAR','SVMRBF','REGLOG','MLP'])
print(resultadocsv)
resultadocsv.to_csv("C:/Users/Auricelia/Desktop/ResultadosReshipe/RESULTCSV.csv")


# implementando o teste de Friedman para todos os algoritmos usados
n, p = stats.friedmanchisquare(acertoK_3,acertoK_5,acertoK_7,acertoKP_3,acertoKP_5,acertoKP_7,acertonaiveBayes,
acertoArvoreDecisao,acertoSVMrbf,acertoSVMLinear,logisticRarray,MLP_acerto)
resultadocsv.boxplot(figsize=(12,8))
plt.title("P-value: " + str(p))
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.savefig("C:/Users/Auricelia/Desktop/ResultadosReshipe/BOXPLOTRESULTADO")
plt.show()

# Terminam os algoritmos de classificação '''
