import base64
import os
import google.generativeai as genai
from google.generativeai import types

from dotenv import load_dotenv
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
   model_name = "gemini-2.0-flash",
   generation_config = generation_config,
)

history = []

print("Bot: Hi, how can I assist you today?")

while True:
    user_input = input("You: ")

    # Prepare conversation history for the model
    contents = []
    for item in history:
        contents.append({"role": item["role"], "parts": [item["content"]]})

    contents.append({"role": "user", "parts": [user_input]})  # Add the current user input

    response = model.generate_content(
        contents=contents
        )

    model_response = response.text

    print(f"Bot: {model_response}")
    print()

    # Update conversation history
    history.append({"role": "user", "content": user_input}) #Add user input to history
    history.append({"role": "model", "content": model_response}) #Add bot response to history
