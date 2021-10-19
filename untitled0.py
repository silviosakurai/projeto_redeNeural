# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import tensorflow as tf
from plotly.offline import plot
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout
import plotly.graph_objs as go


classe = pd.read_csv('D:\POS\TCC\saidas.csv', encoding='latin-1', header=None)

previsores = pd.read_csv('D:\POS\TCC\entradas.csv', encoding='latin-1')
print(previsores)
le = preprocessing.LabelEncoder()

previsores['TUMOR'] = le.fit_transform(previsores['TUMOR'])

output = open('tumor_encoder.pkl', 'wb')
pickle.dump(le, output)
output.close()

previsores['NODULO'] = le.fit_transform(previsores['NODULO'].astype(str))

output = open('nodulo_encoder.pkl', 'wb')
pickle.dump(le, output)
output.close()

previsores['METASTASE'] = le.fit_transform(previsores['METASTASE'].astype(str))

output = open('metastase_encoder.pkl', 'wb')
pickle.dump(le, output)
output.close()

previsores['IDADE'] = le.fit_transform(previsores['IDADE'].astype(str))

output = open('idade_encoder.pkl', 'wb')
pickle.dump(le, output)
output.close()

previsores['SEXO'] = le.fit_transform(previsores['SEXO'].astype(str))

output = open('sexo_encoder.pkl', 'wb')
pickle.dump(le, output)
output.close()

previsores['CID'] = le.fit_transform(previsores['CID'].astype(str))

output = open('cid_encoder.pkl', 'wb')
pickle.dump(le, output)
output.close()

prev_train, prev_test, class_train, class_test = train_test_split(previsores, classe, test_size=0.25)

classificador = Sequential()

classificador.add(Dense(units = 32, activation = 'elu', kernel_initializer='random_uniform', input_dim=6))

classificador.add(Dropout(0.25))

classificador.add(Dense(units = 16, activation = 'elu', kernel_initializer='random_uniform'))

classificador.add(Dropout(0.25))

classificador.add(Dense(units = 8, activation = 'elu', kernel_initializer='random_uniform'))


classificador.add(Dense(units=1, activation='sigmoid'))

otimizador = tf.keras.optimizers.Adam(learning_rate= 0.001, decay = 0.001, clipvalue = 0.1521)

classificador.compile(otimizador, loss='mse', metrics=['binary_accuracy'])

historico = classificador.fit(prev_train, class_train, batch_size=10, epochs=1000)

classificador_json = classificador.to_json()

with open('D:\POS\TCC\silvio_sakurai.json', 'w') as json_file:
    json_file.write(classificador_json)

classificador.save_weights('D:\POS\TCC\silvio_sakurai.h5')

previsoes = classificador.predict(prev_test)
previsoes = previsoes > 0.5

from sklearn.metrics import confusion_matrix, accuracy_score

precisao = accuracy_score(class_test, previsoes)


matriz = confusion_matrix(class_test, previsoes)

resultado = classificador.evaluate(prev_test, class_test)

fig = go.Figure()

fig.add_trace(go.Scattergl(y=historico.history['binary_accuracy'],
                    name='Valid'))

fig.update_layout(height=500, width=1300,
                  xaxis_title='Epoch',
                  yaxis_title='Acurácia')

plot(fig, auto_open=True)

print(resultado)