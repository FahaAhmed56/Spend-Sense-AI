from src.model import predict, train, DATA

def test_predict_single():
    result = predict('Uber trip downtown')
    assert len(result) == 1
    assert result[0]['category']
    assert 0 <= result[0]['confidence'] <= 1

def test_predict_multiple():
    assert len(predict(['Uber ride','Netflix subscription'])) == 2

def test_blank_rejected():
    import pytest
    with pytest.raises(ValueError):
        predict([' '])

def test_train_temporary(tmp_path):
    model = train(DATA, tmp_path / 'model.joblib')
    assert (tmp_path / 'model.joblib').exists()
    assert hasattr(model, 'predict_proba')
