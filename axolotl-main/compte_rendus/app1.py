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
    

if __name__ == '__main__':
    app.run(debug=True)
   
