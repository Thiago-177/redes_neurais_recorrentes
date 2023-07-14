from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 

base = pd.read_csv('petr4_treinamento_ex.csv')
#apagando valores nulos 
base = base.dropna()

base_treinamento = base.iloc[:, 1:2].values #coluna que vamos fazer as previsões

#normlizando
normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_norm = normalizador.fit_transform(base_treinamento)

#preparando este tipo de base de dados.
#1 definimos os intervalo de tempo (exemplo a cada 4 valores ele prediz 1)

previsores90 = [] #armazenará 90 primeiros dados (tempo = 90)
preco_real90 = []

for i in range(90, 1342):
    previsores90.append(base_treinamento_norm[i-90:i, 0])
    preco_real90.append(base_treinamento_norm[i, 0])


previsores90, preco_real90 = np.array(previsores90), np.array(preco_real90)
previsores90 = np.reshape(previsores90, (previsores90.shape[0], previsores90.shape[1], 1))

regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores90.shape[1], 1))) #units = numero de cel de memorias
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50)) # quando for a ultima camada retirar o return_sequences
regressor.add(Dropout(0.3))

regressor.add(Dense(units=1, activation = 'linear')) #linear pois é um caso de regressao
#a função sigmoide também pode ser usada pois os dados estao normalizados entre 0 e 1

regressor.compile(optimizer='rmsprop', loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error'])#rmsprop e o mais indicado para recorrentes pmas o adam também funciona 

regressor.fit(previsores90, preco_real90, epochs = 100, batch_size = 32) #minimo de 100 epocas

base_teste = pd.read_csv('petr4_teste_ex.csv')
preco_real_teste = base_teste.iloc[:,1:2].values
base_completa = pd.concat((base['Open'], base_teste['Open']), axis=0)

entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values 
#vamos comear a buscar os registros a partir de 1152 para o modelo poder ser treinado com as ultimas 90
entradas = entradas.reshape(-1,1)

entradas = normalizador.transform(entradas) # atenção para nao colocar o fit_transform
# o fit o é usado quando vamos normalizar a primeira vez com um conjunto de dados maior 
#no caso o que queremso fazer e normaliazar os dados de entrada e que eles se encaixem na normalização ja 
#feita anteriormente com o Fit transform, por isso utilizamos apenas o "Transform"

X_teste = []
for i in range(90, 109):
    X_teste.append(entradas[i-90:i,0])

X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))

previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes) 
#função muito legal que devolve o inverso da normalização 

previsoes.mean()
preco_real_teste.mean()

plt.plot(preco_real_teste, color='red', label='Preço Real')
plt.plot(previsoes, color = 'blue', label = 'Previsões')
plt.title('Previsão dos Preços das Ações')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()


















