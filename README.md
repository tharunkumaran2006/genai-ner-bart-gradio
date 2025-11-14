## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Identifying key information such as names, locations, and organizations within unstructured text is a challenging task that often requires manual effort. This project aims to build a Named Entity Recognition (NER) web application that automatically detects and highlights entities in any given text using a pre-trained transformer model (dslim/bert-base-NER). The goal is to make text analysis faster, more accurate, and user-friendly through an interactive Gradio interface.

### DESIGN STEPS:

#### STEP 1:
Decide what the app should do — in this case, it identifies named entities (like names, locations, organizations) from text using a Hugging Face NER model (dslim/bert-base-NER).

#### STEP 2:
Use Gradio components to build a clean, user-friendly layout

#### STEP 3:
Configure and Launch the App

### PROGRAM:
```python
import os
import io
import gradio as gr
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']

```
```python
# Helper function
import requests, json

#Summarization endpoint
def get_completion(inputs, parameters=None,ENDPOINT_URL=os.environ['HF_API_SUMMARY_BASE']): 
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL, headers=headers,
                                data=json.dumps(data)
                               )
    return json.loads(response.content.decode("utf-8"))
```
```
API_URL = os.environ['HF_API_NER_BASE'] #NER endpoint
text = "My name is Jagadeesh, i'm studying in Saveetha Engineering College and i lives in Chennai"
get_completion(text, parameters=None, ENDPOINT_URL= API_URL)
```
```python
def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    return {"text": input, "entities": output}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    #Here we introduce a new tag, examples, easy to use examples for your application
                    examples=["My name is Jagadeesh, i'm studying in Saveetha Engineering College and i lives in Chennai"])
demo.launch(share=True, server_port=int(os.environ['PORT3']))
```
```python
def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        label = token["entity"]
        word = token["word"]

        clean_word = word.replace("##", "")

        if merged_tokens:
            prev = merged_tokens[-1]
            prev_label = prev["entity"]

            if label.endswith(prev_label.split("-")[-1]) and (label.startswith("I-") or label.startswith("B-")):
                prev["word"] += clean_word
                prev["end"] = token["end"]
                prev["score"] = (prev["score"] + token["score"]) / 2
                continue

            if word.startswith("##") and prev_label.startswith("B-"):
                prev["word"] += clean_word
                prev["end"] = token["end"]
                prev["score"] = (prev["score"] + token["score"]) / 2
                continue
            
            if prev["entity"].endswith("PER") and len(prev["word"]) == 1 and word.startswith("##"):
                prev["word"] += clean_word
                prev["end"] = token["end"]
                prev["score"] = (prev["score"] + token["score"]) / 2
                continue


        merged_tokens.append({
            "entity": label,
            "word": clean_word,
            "start": token["start"],
            "end": token["end"],
            "score": token["score"]
        })

    return merged_tokens


def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(
    fn=ner,
    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
    outputs=[gr.HighlightedText(label="Text with entities")],
    title="NER with Fine-Tuned BART ",
    description="Highlights people, organizations, and locations in your text.",
    allow_flagging="never",
    examples=[
        ["My name is Tharun V K, I am pursueing Artificial Intelligence and Data Science and I live in Chennai."],
        ["Elon Musk founded SpaceX in the United States"]
    ]
)

demo.launch(share=True, server_port=int(os.environ.get("PORT4", 7860)))
```
### OUTPUT:
<img width="936" height="748" alt="image" src="https://github.com/user-attachments/assets/16eca51f-3baa-45ce-87de-099c762e8bbf" />





### RESULT:
The NER app successfully identifies and highlights entities such as names, locations, and organizations from user input text using the dslim/bert-base-NER model.
