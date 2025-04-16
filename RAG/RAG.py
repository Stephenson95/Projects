# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:17:21 2024

@author: Stephenson
"""

#import getpass
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.document_loaders import WikipediaLoader

#def _set_env(key: str):
#    if key not in os.environ:
#        os.environ[key] = getpass.getpass(f"{key}:")

#key = open(r'C:\Users\Stephenson\Desktop\Code\Keys\nomic_api.txt','r').readlines()[0]
#os.environ['NOMIC_API_KEY'] = key  
#os.environ['USER_AGENT'] = "test_agent"  

#_set_env("NOMIC_API_KEY")
#_set_env("USER_AGENT") #my_agent

#%%
#Create vector database
# urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/",
#         "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
#         "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"]



# docs = [WebBaseLoader(url).load() for url in urls]
# docs_list = [item for sublist in docs for item in sublist]

# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=250, chunk_overlap=0
# )
# doc_splits = text_splitter.split_documents(docs_list)

query = "Naruto Uzumaki"
raw_documents = WikipediaLoader(query=query).load()


# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=raw_documents,
    collection_name="rag-chroma",
    embedding=NomicEmbeddings(model="nomic-embed-text-v1.5",
                              device='gpu',
                              inference_mode="local")
)
retriever = vectorstore.as_retriever()
#%%
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

local_llm = "mistral"

# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()
question = "Orochimaru"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

