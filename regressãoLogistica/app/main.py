from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib

app = FastAPI()

class request_body(BaseModel):
    A_id: int
    Size: float
    Weight: float
    Sweetness: float
    Crunchiness: float
    Juiciness: float
    Ripeness: float
    Acidity: float


modelo_qualidade = joblib.load('model_lr_qualidade_fruta.pkl')

@app.post("/classify")
def predict(data: request_body):

    input_features= [[data.Size, data.Weight, data.Sweetness, data.Crunchiness, data.Juiciness, 
                      data.Ripeness, data.Acidity]]
    
    y_pred = modelo_qualidade.predict(input_features)[0].astype(int)
    y_prob = modelo_qualidade.predict_proba(input_features)[0].astype(float)

    resposta = 'Boa' if y_pred == 1 else 'Ruim'
    probabilidade = y_prob[y_pred]

    return {'qualidade': resposta, 'probabilidade': probabilidade}