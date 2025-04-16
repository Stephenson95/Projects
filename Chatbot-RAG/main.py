import os
from langchain.chat_models import init_chat_model
from langchain_mistralai import MistralAIEmbeddings
from langchain_astradb import AstraDBVectorStore

llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
embeddings = MistralAIEmbeddings(model="text-embedding-3-large")
vector_store = AstraDBVectorStore(
                                    embedding=embeddings,
                                    api_endpoint=os.environ.get("ASTRA_DB_ENDPOINT"),
                                    collection_name=os.environ.get("ASTRA_DB_COLLECTION"),
                                    token=os.environ.get("ASTRA_DB_APPLICATION_TOKEN"),
                                    namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
                                )

