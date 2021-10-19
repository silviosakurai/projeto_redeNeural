# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:02:05 2021

@author: silvi
"""
import tensorflow.keras as keras
import pandas as pd
import pickle
from flask import Flask
from flask_restful import Resource, Api, reqparse

with open("D:\POS\TCC\silvio_sakurai.json") as f:
    model = keras.models.model_from_json(f.read())
    model.load_weights("D:\POS\TCC\silvio_sakurai.h5")

app = Flask(__name__)
api = Api(app)

class RedeNeural(Resource):
    def post(self):
        parser = reqparse.RequestParser()  # initialize
        
        parser.add_argument('TUMOR', required=True)  # add args
        parser.add_argument('NODULO', required=True)
        parser.add_argument('METASTASE', required=True)
        parser.add_argument('IDADE', required=True)
        parser.add_argument('SEXO', required=True)
        parser.add_argument('CID', required=True)


        args = parser.parse_args()  # parse arguments to dictionary
        # create new dataframe containing new values
        
        data = {
            'TUMOR': args['TUMOR'].strip(),
            'NODULO': args['NODULO'].strip(),
            'METASTASE': args['METASTASE'].strip(),
            'IDADE': args['IDADE'].strip(),
            'SEXO': args['SEXO'].strip(),
            'CID': args['CID'].strip(),

        }
        previsores = pd.DataFrame(data=data, index=[0])
        
       
        tumor_encoder = open('tumor_encoder.pkl', 'rb')
        le = pickle.load(tumor_encoder) 
        tumor_encoder.close()
        previsores['TUMOR'] = le.transform(previsores['TUMOR'])
        
        tumor_encoder = open('nodulo_encoder.pkl', 'rb')
        le = pickle.load(tumor_encoder) 
        tumor_encoder.close()
        previsores['NODULO'] = le.transform(previsores['NODULO'])
        
        tumor_encoder = open('metastase_encoder.pkl', 'rb')
        le = pickle.load(tumor_encoder) 
        tumor_encoder.close()
        previsores['METASTASE'] = le.transform(previsores['METASTASE'])
        
        tumor_encoder = open('idade_encoder.pkl', 'rb')
        le = pickle.load(tumor_encoder) 
        tumor_encoder.close()
        previsores['IDADE'] = le.transform(previsores['IDADE'])
        
        tumor_encoder = open('sexo_encoder.pkl', 'rb')
        le = pickle.load(tumor_encoder) 
        tumor_encoder.close()
        previsores['SEXO'] = le.transform(previsores['SEXO'])
        
        tumor_encoder = open('cid_encoder.pkl', 'rb')
        le = pickle.load(tumor_encoder) 
        tumor_encoder.close()
        previsores['CID'] = le.transform(previsores['CID'])
        

        resultado =  round(model.predict(previsores)[0,0])
        print(resultado)
        return {'data': resultado}, 200


api.add_resource(RedeNeural, '/RedeNeural')

if __name__ == '__main__':
    app.run()  # run our Flask app
    
