import streamlit as st
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from llm import llm, embeddings

neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                              # (1)
    url=st.secrets["NEO4J_URI"],             # (2)
    username=st.secrets["NEO4J_USERNAME"],   # (3)
    password=st.secrets["NEO4J_PASSWORD"],   # (4)
    index_name="twitter_embeddings_index",                 # (5)
    node_label="Tweet",                      # (6)
#     text_node_property="name",               # (7)
#     embedding_node_property="nameEmbedding", # (8)
#     retrieval_query="""
# RETURN
#     node.name AS text,
#     score,
#     {
#         followers: node.followers,
#         following: node.following,
#         location: node.location,
#         screen_name: node.screen_name
#     } AS metadata
# """
)

retriever = neo4jvector.as_retriever()

from langchain.chains import RetrievalQA

kg_qa = RetrievalQA.from_chain_type(
    llm,                  # (1)
    chain_type="stuff",   # (2)
    retriever=retriever,  # (3)
)