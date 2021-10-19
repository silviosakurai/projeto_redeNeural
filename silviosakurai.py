
import pandas as pd
import seaborn as sns
from plotly.offline import plot
from sklearn import preprocessing 
import plotly.graph_objs as go

le = preprocessing.LabelEncoder()

previsores = pd.read_csv('D:\POS\TCC\silvio.csv', encoding='latin-1')

previsores['TUMOR'] = le.fit_transform(previsores['TUMOR'])
previsores['NODULO'] = le.fit_transform(previsores['NODULO'])
previsores['METASTASE'] = le.fit_transform(previsores['METASTASE'])
previsores['IDADE'] = le.fit_transform(previsores['IDADE'])
previsores['SEXO'] = le.fit_transform(previsores['SEXO'])
previsores['CID'] = le.fit_transform(previsores['CID'])


from sklearn.model_selection import train_test_split

X = previsores.drop(columns=['I'])
y = previsores.drop(columns=['TUMOR', 'NODULO', 'METASTASE', 'IDADE', 'SEXO', 'CID', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
print(X)
print(y)
prev_train, prev_test, class_train, class_test = train_test_split(X, y, test_size=0.3)

#vendo a correlação entre as variáveis
a = previsores.corr()

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout

classificador = Sequential()

classificador.add(Dense(units = 32, activation = 'sigmoid', kernel_initializer='normal', input_dim=14))

classificador.add(Dense(units=1,activation='relu'))

otimizador = keras.optimizers.Adam()

classificador.compile(otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])

history = classificador.fit(prev_train, class_train, batch_size=50, epochs=1500)


#plot do grafico da curva de aprendizado no treino
fig = go.Figure()

fig.add_trace(go.Scattergl(y=history.history['binary_accuracy'],
                    name='Valid'))

fig.update_layout(height=500, width=700,
                  xaxis_title='Epoch',
                  yaxis_title='MAE')

plot(fig, auto_open=True)

#calculando as previsões
previsoes = classificador.predict(prev_test)

from sklearn.metrics import mean_absolute_error

#utilizando o MAE para ver a precisão
precisao = mean_absolute_error(class_test, previsoes)

resultado = classificador.evaluate(prev_test, class_test)

print(precisao)
print(resultado)


