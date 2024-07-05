from pydantic import BaseModel
from fastapi import FastAPI
import joblib
import numpy as np

# Instanciando API
app = FastAPI()

# Definindo estrutura do corpo da requisição
class request_body(BaseModel):
    horas_estudo : float

# Carregando modelo
modelo_pontuacao = joblib.load('modelo_reg_log.pkl')

@app.post('/predict')
def predicao(data : request_body):
    
    input_feature = [[data.horas_estudo]]

    # Realizando predição
    y_pred = modelo_pontuacao.predict(input_feature)[0].astype(int)

    return {'pontuacao': y_pred.tolist()}
