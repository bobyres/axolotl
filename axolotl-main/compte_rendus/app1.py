import subprocess
import time
from flask import Flask, request, jsonify
from infer import chat  # Importez votre script d'inférence
import requests

app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    data = request.get_json()
    query = data.get('query', '')
    history = data.get('history', [])
    temperature = data.get('temperature', 0.7)

    response, updated_history = chat(query, history=history, temperature=temperature)

    return jsonify({'response': response, 'updated_history': updated_history})

def run_flask_app():
    app.run(debug=True)

if __name__ == '__main__':
    # Démarrer le serveur Flask dans un processus distinct
    flask_process = subprocess.Popen(['python', 'app.py', 'run_flask_app'])

    # Attendre un certain temps pour que le serveur Flask démarre
    time.sleep(2)

    # Effectuer la requête
    url = 'http://127.0.0.1:5000/inference'
    data = {'query': 'Votre question ici', 'history': ['Historique 1', 'Historique 2'], 'temperature': 0.7}
    response = requests.post(url, json=data)

    print(response.json())
    time.sleep(15)
    # Terminer le processus Flask après utilisation
    flask_process.terminate()
