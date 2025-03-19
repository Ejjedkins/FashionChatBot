import base64
import os
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import re

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = types.GenerationConfig(
    temperature=2,
    top_p=0.95,
    top_k=40,
    max_output_tokens=8192,
    response_mime_type="text/plain",
)

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
)

app = Flask(__name__)
CORS(app)

history = []
image_history = []


def format_response(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
    text = re.sub(r'\\boxed\{(.*?)\}', r'\1', text)  # Remove \boxed{...}
    text = re.sub(r'\\sqrt\{(.*?)\}', r'sqrt(\1)', text)  # Replace \sqrt{...}
    text = re.sub(r'\\cdot', r'×', text)  # Replace \cdot with ×
    text = re.sub(r'\$', r'', text) #remove dollar signs

    # Handle bullets and indentation
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        if line.strip().startswith('- '):  # Bullet point
            formatted_lines.append(f'<li>{line.strip()[2:]}</li>')
        elif line.startswith('    '):  # Indentation (4 spaces)
            formatted_lines.append(f'<div style="margin-left: 20px;">{line.strip()}</div>')
        else:
            formatted_lines.append(line)

    formatted_text = '\n'.join(formatted_lines)

    if any(line.startswith('<li>') for line in formatted_lines):
        formatted_text = "<ul>\n" + formatted_text + "\n</ul>"

    return formatted_text

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    image_data = request.json.get('image')

    contents = []

    for item in history:
        if "image_data" in item:
            contents.append({"role": item["role"], "parts": [item["content"], {"mime_type": "image/jpeg", "data": base64.b64decode(item["image_data"].split(',')[1])}]})
        else:
            contents.append({"role": item["role"], "parts": [item["content"]]})

    prompt = user_input
    if image_data:
        # When image data is present, only include the current user input and image in the contents
        prompt = f"""You are an outfit recommendation chatbot. 
        You will be given an image of an outfit and a question about whether 
        it is suitable for a specific occasion. Analyze the image and determine 
        if the outfit is appropriate for the occasion described in the following question: '{user_input}'.

Provide a concise 'yes' or 'no' answer followed by a brief explanation if asked if you would recommend or suggest 
the outfit given a specific occasion. You can also tell the user what clothes are in the picture if asked to do so.
"""
        image_parts = [{"mime_type": "image/jpeg", "data": base64.b64decode(image_data.split(',')[1])}]
        contents = [{"role": "user", "parts": [prompt, image_parts[0]]}]
        # We are not adding to history or image_history here for this specific image-based request
    else:
        # For text-only messages, maintain the history
        for item in history:
            contents.append({"role": item["role"], "parts": [item["content"]]})
        contents.append({"role": "user", "parts": [user_input]})
        history.append({"role": "user", "content": user_input})

    response = model.generate_content(contents=contents)
    model_response = response.text
    model_response = format_response(model_response)

    # Update history only for text-based interactions
    if not image_data:
        history.append({"role": "model", "content": model_response})

    return jsonify({'response': model_response})

@app.route('/style.css')
def serve_css():
    return send_from_directory('.', 'style.css')


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
