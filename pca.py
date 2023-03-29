# Personal Chat Appliance
# version 0.1 by @mattesmattes

from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, Document
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os
import codecs
from unidecode import unidecode


os.environ["OPENAI_API_KEY"] = '_Your API Key_'

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 2048
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))

    documents = []
    for filename in os.listdir(directory_path):
        with open(os.path.join(directory_path, filename), "rb") as f:
            data = f.read().decode("utf-8", "replace")
            # Replace non-ASCII characters with their closest ASCII equivalent
            data = unidecode(data)
            document = Document(data)
            document.metadata = {"filename": filename}
            documents.append(document)

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Your Personal Chat Appliance")

index = construct_index("docs")
iface.launch(share=True)
