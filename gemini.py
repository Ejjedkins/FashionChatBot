import base64
import os
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv
from flask import Flask, request, jsonify
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

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    image_data = request.json['image']

    contents = []
    for item in history:
        if "image_data" in item:
            contents.append({"role": item["role"], "parts": [item["content"], {"mime_type": "image/jpeg", "data": base64.b64decode(item["image_data"].split(',')[1])}]})
        else:
            contents.append({"role": item["role"], "parts": [item["content"]]})

    prompt = user_input
    if image_data:
        # Refined prompt
        prompt = f"""You are an outfit recommendation chatbot. You will be given an image and a question about the suitability of the outfit IN THE IMAGE for a specific event. Follow these steps:

1. Analyze the image and identify the clothing items.
2. Describe the typical weather conditions for the event described in the question.
3. Determine if the OVERALL OUTFIT IN THE IMAGE is suitable for the event's weather.
4. If the outfit is suitable, explain why. If the outfit is not suitable, explain why not.
5. Provide a clear 'yes' or 'no' answer to the question: '{user_input}'

Here is the question: '{user_input}'
Please provide your answer in a human-readable format, not as raw bounding box detections.
"""

        image_parts = [{"mime_type": "image/jpeg", "data": base64.b64decode(image_data.split(',')[1])}]
        contents.append({"role": "user", "parts": [prompt, image_parts[0]]})
        image_history.append({"role": "user", "content": prompt, "image_data": image_data})
    else:
        contents.append({"role": "user", "parts": [prompt]})
        history.append({"role": "user", "content": user_input})

    response = model.generate_content(contents=contents)

    model_response = response.text

    model_response = format_response(model_response)

    history.append({"role": "model", "content": model_response})

    return jsonify({'response': model_response})

if __name__ == '__main__':
    app.run(debug=True)

    # Update conversation history
    history.append({"role": "user", "content": user_input}) #Add user input to history
    history.append({"role": "model", "content": model_response}) #Add bot response to history
