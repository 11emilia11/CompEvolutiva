import pandas as pd
import FuncoesML as fun
from scipy import stats
import numpy as np
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
reshipe = open('C:/Users/Auricelia/Documents/Ciencia da Computacao/Computação Evolutiva/Base de dados/ReshibaseQ.txt',
               "rt", encoding="utf8")

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


#criando os vetores que irão receber os ingredientes de cada classe
ingredientes_1 = []
ingredientes_2 = []
ingredientes_3 = []
ingredientes_4 = []

#preenchendo o vetor 'todosingredientes' sem repetição
for rec in receitas:
    for ingrediente in rec.ingredientes:

        if rec.Class == '1':
            if ingredientes_1.__contains__(ingrediente) == False:
                ingredientes_1.append(ingrediente)
        elif rec.Class == '2':
            if ingredientes_2.__contains__(ingrediente) == False:
                ingredientes_2.append(ingrediente)
        elif rec.Class == '3':
            if ingredientes_3.__contains__(ingrediente) == False:
                ingredientes_3.append(ingrediente)
        elif rec.Class == '4':
            if ingredientes_4.__contains__(ingrediente) == False:
                ingredientes_4.append(ingrediente)

print('ingr classe 1')
print(ingredientes_1)
print()

print('ingr classe 2')
print(ingredientes_2)
print()

print('ingr classe 3')
print(ingredientes_3)
print()

print('ing classe 4')
print(ingredientes_4)
print()


todosingredientes = []
#preenchendo o vetor 'todosingredientes' sem repetição
for rec in receitas:
    for ingrediente in rec.ingredientes:
        if todosingredientes.__contains__(ingrediente) == False:
            todosingredientes.append(ingrediente)



#ordenando o vetor
todosingredientes = sorted(todosingredientes)

#ordenando os vetores
ingredientes_1 = sorted(ingredientes_1)
ingredientes_2 = sorted(ingredientes_2)
ingredientes_3 = sorted(ingredientes_3)
ingredientes_4 = sorted(ingredientes_4)

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


print('entrada')
print(ingredientes_1)
print()

print('prato principal')
print(ingredientes_2)
print()

print('acompanhamento')
print(ingredientes_3)
print()

print('sobremesa')
print(ingredientes_4)
print()


#separando as receitas por classe
df_1 = df.loc[df['Class'] == '1']
df_2 = df.loc[df['Class'] == '2']
df_3 = df.loc[df['Class'] == '3']
df_4 = df.loc[df['Class'] == '4']

print('df 1')
print(df_1.head())
print(df_1.shape[0])
print()
print('df 2')
print(df_2.head())
print()
print('df 3')
print(df_3.head())
print()
print('df 4')
print(df_4)
print()


# criando a população de receitas com 20 cromossomos de tamanho 38
# cromossomos = fun.create_population(20, tamanho_ingredientes)
# for cromo in cromossomos:


# j = 0
# y = df[j:j+1]
# df_2.append(y)
# print(df.at[7, 'Class'])
# print(df[0:1])
# print('adicionando 1 linha ao df_2')
# print(df_2)

# print(df.loc[df['Class']== '1'])

#criando os vetores que irão armazenar as probabilidades de cada ingrediente por classe
p_ingrediente_1 = []
p_ingrediente_2 = []
p_ingrediente_3 = []
p_ingrediente_4 = []

# gerando as probabilidades para a classe '1'
total_rec_1 = df_1.shape[0]
df_1_noclass = df_1.copy()
del df_1_noclass['Class']

for i in range(total_rec_1):

    soma = df_1_noclass[i].sum()
    prob_ingrediente = soma / total_rec_1
    p_ingrediente_1.append(prob_ingrediente)

print(p_ingrediente_1)

# gerando as probabilidades para a classe '2'

total_rec_2 = df_2.shape[0]
df_2_noclass = df_2.copy()
del df_2_noclass['Class']

for j in range(total_rec_2):
    soma = df_2_noclass[j].sum()
    prob_ingrediente = soma / total_rec_2
    p_ingrediente_2.append(prob_ingrediente)

print(p_ingrediente_2)

# gerando as probabilidades para a classe '3'

total_rec_3 = df_3.shape[0]
df_3_noclass = df_3.copy()
del df_3_noclass['Class']

for k in range(total_rec_3):
    soma = df_3_noclass[k].sum()
    prob_ingrediente = soma / total_rec_3
    p_ingrediente_3.append(prob_ingrediente)

print(p_ingrediente_3)

# gerando as probabilidades para a classe '4'

total_rec_4 = df_4.shape[0]
df_4_noclass = df_4.copy()
del df_4_noclass['Class']

for l in range(total_rec_3):
    soma = df_4_noclass[l].sum()
    prob_ingrediente = soma / total_rec_4
    p_ingrediente_4.append(prob_ingrediente)


print(p_ingrediente_4)

tamanho = 402
cromossomos = fun.create_population(2,404)
print()
print(cromossomos)



