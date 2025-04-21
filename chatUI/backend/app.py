import os
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}}) 


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No OPENAI_API_KEY set for Flask application")

client = openai.OpenAI(api_key=api_key)

SYSTEM_PROMPT = {"role": "system", "content": "You are a helpful assistant."}

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message_history = data.get('history')

        if not message_history or not isinstance(message_history, list):
            return jsonify({"error": "Invalid or missing message history"}), 400
        messages_to_send = [SYSTEM_PROMPT] + message_history

        completion = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages_to_send
        )

        bot_response = completion.choices[0].message.content

        return jsonify({"reply": bot_response})

    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        return jsonify({"error": f"OpenAI API Error: {e}"}), 500
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

