from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.model import predict

app = FastAPI(title='SpendSense AI', version='1.0.0')

class PredictRequest(BaseModel):
    descriptions: list[str] = Field(min_length=1, max_length=100)

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict')
def classify(req: PredictRequest):
    if any(not x.strip() for x in req.descriptions):
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail='Descriptions must not be blank')
    return {'results': predict(req.descriptions)}
