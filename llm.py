import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq



# llm = ChatOpenAI(
#     openai_api_key=st.secrets["OPENAI_API_KEY"],
#     model=st.secrets["OPENAI_MODEL"],
# )

llm = ChatGroq(temperature=0, groq_api_key=st.secrets["OPENAI_API_KEY"], model_name="mixtral-8x7b-32768")

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

# from langchain_openai import OpenAIEmbeddings

# embeddings = OpenAIEmbeddings(
#     openai_api_key=st.secrets["OPENAI_API_KEY"]
# )


# import os

# os.environ["NLPCLOUD_API_KEY"] = "9ff7a2476f8e0c3cca3f21311ce32594b94667b3"
# from langchain.chains import LLMChain
# from langchain_community.llms import NLPCloud
# from langchain_community.embeddings import NLPCloudEmbeddings
# from langchain_core.prompts import PromptTemplate

# llm = NLPCloud()

# llm_chain = LLMChain(prompt=prompt, llm=llm)

# embeddings = NLPCloudEmbeddings(
# )