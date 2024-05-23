import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import dotenv
import os


dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


llm = OpenAI(api_key=OPENAI_API_KEY)


template = """
Vous êtes un conseiller d'orientation. Vous devez répondre aux questions des utilisateurs concernant l'orientation scolaire et professionnelle. 
Si la question n'est pas en rapport avec l'orientation, répondez simplement que vous êtes un conseiller en orientation.
chat_history : {chat_history}
Human: {input}
AI:
"""

prompt = PromptTemplate(input_variables=["chat_history", "input"], template=template)


memory = ConversationBufferMemory(memory_key="chat_history", k=5)


conversation = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory = memory
)


st.set_page_config(
    page_title = "Orientation BOT",
    page_icon = "",
    layout = "wide"
)

st.title("Orientation BOT")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content" : "Salut posez moi votre question...."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({"role":"user ", "content":user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Chargement....."):
            response = conversation.predict(input = user_prompt)
            st.write(response)
    new_message = {"role": "assistant", "content": response}
    st.session_state.messages.append(new_message)