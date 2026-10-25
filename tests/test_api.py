from fastapi.testclient import TestClient
from api.main import app
client=TestClient(app)

def test_health():
    assert client.get('/health').json()['status']=='ok'

def test_prediction():
    res=client.post('/predict',json={'descriptions':['Uber ride airport']})
    assert res.status_code==200
    assert 'category' in res.json()['results'][0]

def test_invalid():
    assert client.post('/predict',json={'descriptions':['']}).status_code==422
