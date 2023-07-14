from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

base = pd.read_csv('petr4_treinamento.csv')
base = base.dropna()

base_treinamento = base.iloc[:, 1:7].values

normalizador = MinMaxScaler(feature_range = (0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

previsores = []
preco_real = []

for i in range(90,1242):
    previsores.append(base_treinamento_normalizada[i-90:i , 0:6])
    preco_real.append(base_treinamento_normalizada[i, 0])

previsores, preco_real = np.array(previsores), np.array(preco_real)

regressor = Sequential()

regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 6)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 1, activation = 'sigmoid')) # porque os valores estao entre 0 e 1 (normlizados)

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', 
                  metrics = ['mean_absolute_error'])

es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 10, verbose = 1)
#parametro monitor é usado para escolher qual parametro do treianmento sera monitorado
# min delta e o intervalo minimo para ser considerado melhora
#patience é o numero de epocas a se tolerar sem melhoria do eresultado
#verbose é para mostrar na tela o avanço do treinamento

rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1)
#reduz a metrica de aprendizagem quando a metrica para de melhorar
#factor é o valor que a learning rate será multiplicada 

# o proximo será um recurso que salva o modelo a cada época (criando os checkpoints)
mcp = ModelCheckpoint(filepath = 'modelos_salvos/pesos.h5', monitor = 'loss', 
                      save_best_only=True, verbose = 1)

regressor.fit(previsores, preco_real, epochs=100, batch_size = 32, 
              callbacks=(es, rlr, mcp))


















