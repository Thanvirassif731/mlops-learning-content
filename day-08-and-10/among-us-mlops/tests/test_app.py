import pytest
from app import app


def test_health():
    client = app.test_client()
    response = client.get('/health')
    assert response.status_code == 200
    body = response.get_json()
    assert body['status'] == 'ok'
    assert 'Among Us MLOps' in body['message']


def test_model_info():
    client = app.test_client()
    response = client.get('/model-info')
    assert response.status_code == 200
    body = response.get_json()
    assert body['status'] == 'success'
    assert 'model_version' in body
    assert 'features' in body
    assert isinstance(body['features'], list)


def test_predict():
    client = app.test_client()
    response = client.post('/predict', json={
        'Team': 'Crewmate',
        'Task Completed': 5,
        'Imposter Kills': 0,
        'Game Length Sec': 600
    })
    assert response.status_code == 200
    body = response.get_json()
    assert body['status'] == 'success'
    assert 'survival_percentage' in body
    assert 'predicted_sabotages_fixed' in body
    assert 'confidence_band' in body
