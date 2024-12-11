# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 23:37:50 2024

@author: Stephenson
"""
import os
from langchain_community.document_loaders import WikipediaLoader
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings


query = "Naruto Uzumaki"
raw_documents = WikipediaLoader(query=query).load()

# create doc_text as list to store text
doc_text = [d.page_content for d in raw_documents]

# load Diffbot Token for later entity and relationship extraction
diffbot_api_key = "Diffbot-token"
os.environ["DIFFBOT_API_KEY"] = diffbot_api_key
diffbot_nlp = DiffbotGraphTransformer(diffbot_api_key=diffbot_api_key)

# use Diffbot Graph Transformer to extract entities/relationships from Wikipedia articles
graph_documents = diffbot_nlp.convert_to_graph_documents(raw_documents)

# connect to our neo4j database
url = "neo4j-url"
username = "neo4j"
password = "database-pw"

graph = Neo4jGraph(url=url, username=username, password=password)

# import extracted entities and relationships into neo4j graph database
graph.add_graph_documents(graph_documents)


# import LLM
os.environ["OPENAI_API_KEY"] = 'openai-api-key'
os.environ["ANTHROPIC_API_KEY"] = 'anthropic-api-key'
embd = OpenAIEmbeddings()

# from langchain_openai import ChatOpenAI

# model = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(temperature=0, model="claude-3-opus-20240229")