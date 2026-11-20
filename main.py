from fastapi import FastAPI
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
from bs4 import BeautifulSoup
import re
from langchain_core.documents import Document
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

@app.get("/")
def home():
    return {"message": "FastAPI is running!"}


chanks = []
def get_chunk(text, chapter, chunk_size=500, overlap=100, ):
  for i in range(0, len(text), chunk_size-overlap):
    chanks.append((text[i:i+chunk_size], {"chapter" : chapter}))
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
}
urls = [
    "https://bn.wikisource.org/wiki/%E0%A6%AE%E0%A7%87%E0%A6%9C%E0%A6%A6%E0%A6%BF%E0%A6%A6%E0%A6%BF_(%E0%A6%B6%E0%A6%B0%E0%A7%8E%E0%A6%9A%E0%A6%A8%E0%A7%8D%E0%A6%A6%E0%A7%8D%E0%A6%B0_%E0%A6%9A%E0%A6%9F%E0%A7%8D%E0%A6%9F%E0%A7%8B%E0%A6%AA%E0%A6%BE%E0%A6%A7%E0%A7%8D%E0%A6%AF%E0%A6%BE%E0%A6%AF%E0%A6%BC,_%E0%A7%A7%E0%A7%AF%E0%A7%AC%E0%A7%A9)/%E0%A6%AE%E0%A7%87%E0%A6%9C%E0%A6%A6%E0%A6%BF%E0%A6%A6%E0%A6%BF/%E0%A6%8F%E0%A6%95",
    "https://bn.wikisource.org/wiki/%E0%A6%AE%E0%A7%87%E0%A6%9C%E0%A6%A6%E0%A6%BF%E0%A6%A6%E0%A6%BF_(%E0%A6%B6%E0%A6%B0%E0%A7%8E%E0%A6%9A%E0%A6%A8%E0%A7%8D%E0%A6%A6%E0%A7%8D%E0%A6%B0_%E0%A6%9A%E0%A6%9F%E0%A7%8D%E0%A6%9F%E0%A7%8B%E0%A6%AA%E0%A6%BE%E0%A6%A7%E0%A7%8D%E0%A6%AF%E0%A6%BE%E0%A6%AF%E0%A6%BC,_%E0%A7%A7%E0%A7%AF%E0%A7%AC%E0%A7%A9)/%E0%A6%AE%E0%A7%87%E0%A6%9C%E0%A6%A6%E0%A6%BF%E0%A6%A6%E0%A6%BF/%E0%A6%A6%E0%A7%81%E0%A6%87",
    "https://bn.wikisource.org/wiki/%E0%A6%AE%E0%A7%87%E0%A6%9C%E0%A6%A6%E0%A6%BF%E0%A6%A6%E0%A6%BF_(%E0%A6%B6%E0%A6%B0%E0%A7%8E%E0%A6%9A%E0%A6%A8%E0%A7%8D%E0%A6%A6%E0%A7%8D%E0%A6%B0_%E0%A6%9A%E0%A6%9F%E0%A7%8D%E0%A6%9F%E0%A7%8B%E0%A6%AA%E0%A6%BE%E0%A6%A7%E0%A7%8D%E0%A6%AF%E0%A6%BE%E0%A6%AF%E0%A6%BC,_%E0%A7%A7%E0%A7%AF%E0%A7%AC%E0%A7%A9)/%E0%A6%AE%E0%A7%87%E0%A6%9C%E0%A6%A6%E0%A6%BF%E0%A6%A6%E0%A6%BF/%E0%A6%A4%E0%A6%BF%E0%A6%A8",
    "https://bn.wikisource.org/wiki/%E0%A6%AE%E0%A7%87%E0%A6%9C%E0%A6%A6%E0%A6%BF%E0%A6%A6%E0%A6%BF_(%E0%A6%B6%E0%A6%B0%E0%A7%8E%E0%A6%9A%E0%A6%A8%E0%A7%8D%E0%A6%A6%E0%A7%8D%E0%A6%B0_%E0%A6%9A%E0%A6%9F%E0%A7%8D%E0%A6%9F%E0%A7%8B%E0%A6%AA%E0%A6%BE%E0%A6%A7%E0%A7%8D%E0%A6%AF%E0%A6%BE%E0%A6%AF%E0%A6%BC,_%E0%A7%A7%E0%A7%AF%E0%A7%AC%E0%A7%A9)/%E0%A6%AE%E0%A7%87%E0%A6%9C%E0%A6%A6%E0%A6%BF%E0%A6%A6%E0%A6%BF/%E0%A6%9A%E0%A6%BE%E0%A6%B0",
    "https://bn.wikisource.org/wiki/%E0%A6%AE%E0%A7%87%E0%A6%9C%E0%A6%A6%E0%A6%BF%E0%A6%A6%E0%A6%BF_(%E0%A6%B6%E0%A6%B0%E0%A7%8E%E0%A6%9A%E0%A6%A8%E0%A7%8D%E0%A6%A6%E0%A7%8D%E0%A6%B0_%E0%A6%9A%E0%A6%9F%E0%A7%8D%E0%A6%9F%E0%A7%8B%E0%A6%AA%E0%A6%BE%E0%A6%A7%E0%A7%8D%E0%A6%AF%E0%A6%BE%E0%A6%AF%E0%A6%BC,_%E0%A7%A7%E0%A7%AF%E0%A7%AC%E0%A7%A9)/%E0%A6%AE%E0%A7%87%E0%A6%9C%E0%A6%A6%E0%A6%BF%E0%A6%A6%E0%A6%BF/%E0%A6%AA%E0%A6%BE%E0%A6%81%E0%A6%9A",
    "https://bn.wikisource.org/wiki/%E0%A6%AE%E0%A7%87%E0%A6%9C%E0%A6%A6%E0%A6%BF%E0%A6%A6%E0%A6%BF_(%E0%A6%B6%E0%A6%B0%E0%A7%8E%E0%A6%9A%E0%A6%A8%E0%A7%8D%E0%A6%A6%E0%A7%8D%E0%A6%B0_%E0%A6%9A%E0%A6%9F%E0%A7%8D%E0%A6%9F%E0%A7%8B%E0%A6%AA%E0%A6%BE%E0%A6%A7%E0%A7%8D%E0%A6%AF%E0%A6%BE%E0%A6%AF%E0%A6%BC,_%E0%A7%A7%E0%A7%AF%E0%A7%AC%E0%A7%A9)/%E0%A6%AE%E0%A7%87%E0%A6%9C%E0%A6%A6%E0%A6%BF%E0%A6%A6%E0%A6%BF/%E0%A6%9B%E0%A6%AF%E0%A6%BC",
    "https://bn.wikisource.org/wiki/%E0%A6%AE%E0%A7%87%E0%A6%9C%E0%A6%A6%E0%A6%BF%E0%A6%A6%E0%A6%BF_(%E0%A6%B6%E0%A6%B0%E0%A7%8E%E0%A6%9A%E0%A6%A8%E0%A7%8D%E0%A6%A6%E0%A7%8D%E0%A6%B0_%E0%A6%9A%E0%A6%9F%E0%A7%8D%E0%A6%9F%E0%A7%8B%E0%A6%AA%E0%A6%BE%E0%A6%A7%E0%A7%8D%E0%A6%AF%E0%A6%BE%E0%A6%AF%E0%A6%BC,_%E0%A7%A7%E0%A7%AF%E0%A7%AC%E0%A7%A9)/%E0%A6%AE%E0%A7%87%E0%A6%9C%E0%A6%A6%E0%A6%BF%E0%A6%A6%E0%A6%BF/%E0%A6%B8%E0%A6%BE%E0%A6%A4",
    "https://bn.wikisource.org/wiki/%E0%A6%AE%E0%A7%87%E0%A6%9C%E0%A6%A6%E0%A6%BF%E0%A6%A6%E0%A6%BF_(%E0%A6%B6%E0%A6%B0%E0%A7%8E%E0%A6%9A%E0%A6%A8%E0%A7%8D%E0%A6%A6%E0%A7%8D%E0%A6%B0_%E0%A6%9A%E0%A6%9F%E0%A7%8D%E0%A6%9F%E0%A7%8B%E0%A6%AA%E0%A6%BE%E0%A6%A7%E0%A7%8D%E0%A6%AF%E0%A6%BE%E0%A6%AF%E0%A6%BC,_%E0%A7%A7%E0%A7%AF%E0%A7%AC%E0%A7%A9)/%E0%A6%AE%E0%A7%87%E0%A6%9C%E0%A6%A6%E0%A6%BF%E0%A6%A6%E0%A6%BF/%E0%A6%86%E0%A6%9F"
]

for index, url in enumerate(urls):
  response = requests.get(url, headers=headers)
  print(response.status_code)
  response.raise_for_status()

  soup = BeautifulSoup(response.text, "html.parser")
  para = soup.find_all("p")
  t = ""
  for p in para:
      t = t + (p.get_text(strip=True))
  t = t.replace("\n", " ")
  t = re.sub(r'\\u[0-9a-fA-F]{4}|\u200c', ' ', t)
  get_chunk(t, index+1)

llm = ChatOpenRouter(
    model="nvidia/nemotron-3.5-lightning:free",
    temperature=0.7,
    api_key = ""
)

documents = [Document(page_content=text, metadata=meta) for text,meta in chanks]
embedding_model = HuggingFaceEmbeddings(model_name="l3cube-pune/bengali-sentence-similarity-sbert")
vector_store = FAISS.from_documents(documents, embedding_model)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(""" You have to write everything in bangla.
Answer the question using only the following context and also mention the chapter.

Context:
{context}

Question:
{question}

If you find the answer first mention the answer in bangla and then mention the chapter and cite the lines which provide the answer. Everything should be written in bangla.

If the answer is not available in the context, say you don't know.

""")

chain = prompt | llm

def qs_and_ans(ques):
  similar_docs = vector_store.similarity_search(ques, k=3)
  context = similar_docs

  response = chain.invoke({
      "context": context,
      "question": ques
  })
  return response.content




@app.get("/ask")
def ask_question(question: str):
    answer = qs_and_ans(question)
    print(answer)
    return {"response": answer}






@app.get("/chat")
def chat(message: str):
    return {"response": f"You said: {message}"}