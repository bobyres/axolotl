# app.py
from flask import Flask, request, jsonify
from infer import chat  # Importez votre script d'inférence

app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    data = request.get_json()
    query = data.get('query', '')
    history = data.get('history', [])
    temperature = float(data.get('temperature', 0.7))  # Exemple de paramètre, ajustez selon vos besoins

    response, updated_history = chat(query, history=history, temperature=temperature)

    return jsonify({'response': response, 'updated_history': updated_history})

if __name__ == '__main__':
    app.run(debug=True)
