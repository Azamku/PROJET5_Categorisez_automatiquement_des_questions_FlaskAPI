import pytest
from flask_app import app
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("chemin du de tests dans test_flask:", sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))))


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_homepage(client):
    """Test que la page d'accueil retourne un code 200"""
    response = client.get('/')
    assert response.status_code == 200

def test_webhook(client):
    """Test du webhook avec une requête POST valide"""
    response = client.post('/webhook/', json={"ref": "refs/heads/main"})
    assert response.status_code == 200
    assert b"Deployed successfully" in response.data
