import pandas as pd
import keras
import numpy as np
import joblib
from keras.models import Sequential
from sklearn import preprocessing 
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


previsores = pd.read_csv('D:\POS\TCC\silvio.csv', encoding='latin-1')

le = preprocessing.LabelEncoder()


previsores['TUMOR'] = le.fit_transform(previsores['TUMOR'])
previsores['NODULO'] = le.fit_transform(previsores['NODULO'])
previsores['METASTASE'] = le.fit_transform(previsores['METASTASE'])
previsores['IDADE'] = le.fit_transform(previsores['IDADE'])
previsores['SEXO'] = le.fit_transform(previsores['SEXO'])
previsores['CID'] = le.fit_transform(previsores['CID'])
previsores.drop(columns=['I'])

classe = pd.read_csv('D:\POS\TCC\saidas.csv', encoding='latin-1', header=None)

# Agora iremos passar parâmetros para nossa rede
def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()
    # Agora os parâmentos do classificador estão dentro da chamada da função
    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=15))
    classificador.add(Dropout(0.2)) # pega 20% dos neurônios da camada de entrada e zera os valores
    
    
    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    classificador.add(Dropout(0.2))
    
    # Não faz sentido agora alterar a camada de saída já que temos apenas um neurônio
    classificador.add(Dense(units=1, activation='sigmoid'))
    
    # Removi o otimizador do exercício anterior e passei agora como parâmento, o mesmo para o loss
    classificador.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])
    return classificador

#============ Executa a função para ver se está tudo correto ======================
    
classificador = KerasClassifier(build_fn=criarRede)

# Parâmetros são passados através de dicionários
parametros = {'batch_size':[10,30], # valores para o calculo do gradiente descendente
              'epochs': [5, 10], # em caso de utilização em experimentos reais, a quantidade de ópoca precisa ser muito maior
              'optimizer': ['adam', 'sgd'],
              'loss': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [16, 8]}

grid_search = GridSearchCV(estimator=classificador, # passamos o classificador
                           param_grid=parametros, # passamos a grade de parâmetros que acabamos de criar
                           scoring='accuracy',
                           cv = 5) # numero de folds
grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_ 
melhor_precisao = grid_search.best_score_


json = grid_search.model.to_json()
open('json.json', 'w').write(json)

grid_search.model.save_weights('model.h5')
print(melhores_parametros)
print(melhor_precisao)

