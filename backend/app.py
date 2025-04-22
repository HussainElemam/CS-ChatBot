from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from src.retrieve_data import query_rag


load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",  # Allow all origins in development
        "methods": ["GET", "POST", "OPTIONS"],  # Allow these HTTP methods
        "allow_headers": ["Content-Type", "Authorization"],  # Allow these headers
        "supports_credentials": True  # Allow credentials
    }
})


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    print("data", data)
    bot_response = query_rag(data.get('message').get('text'), data.get('history'))

    return jsonify({"reply": bot_response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

