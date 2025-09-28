from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

st.header('Research Tool')
user_input=st.text_input('enter yout prompt')
if st.button:
    st.text('some random text')

