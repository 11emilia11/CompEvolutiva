import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import FuncoesML as fun
from scipy import stats
import numpy as np
from random import randint
import time


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
reshipe = open('C:/Users/Auricelia/Documents/Ciencia da Computacao/Computação Evolutiva/Base de dados/ReshibaseQ.txt'
               , "rt", encoding="utf8")

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
df.to_csv('C:/Users/Auricelia/Documents/Ciencia da Computacao/Computação Evolutiva/Base de dados/Receitas_norm.csv')

print('TAMANHO DF :',df_final[0].size*4)

'''
#gerando a população inicial de cromossomos
cromossomos = fun.create_population(200, 402)
'''


df_1 = df.loc[df['Class'] == '1']
df_2 = df.loc[df['Class'] == '2']
df_3 = df.loc[df['Class'] == '3']
df_4 = df.loc[df['Class'] == '4']
print(df_1)

