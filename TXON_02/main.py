import streamlit as st
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer 
import json
import spacy
nlp = spacy.load('en')
st.set_page_config(layout="wide")
def get_text():
    input_text = st.text_input("You: ",placeholder="Whats on your mind ?")
    return input_text
data = json.loads(open('C:/Users/Sahil Raghuvanshi/Desktop/Projects/ChatBot/data/data_tolokers.json','r').read())
data2 = json.loads(open('C:/Users/Sahil Raghuvanshi/Desktop/Projects/ChatBot/data/sw.json','r').read())
train=[]
for k, row in enumerate(data):
    train.append(row['dialog'][0]['text'])
for k, row in enumerate(data2):
    train.append(row['dialog'][0]['text'])
st.sidebar.image("./data/icon.png")
st.sidebar.title("Hello I am your Machine Learning Chatbot !")
import streamlit.components.v1 as com
with open("style.css") as source:
    design=source.read()
com.html(f"""<style>{design}</style><div class="heading">Machine Learning ChatBot </div>
         <p>I am always ready to help you, Inilialize me by clicking the Initialize Bot button</p>""")
bot = ChatBot(name = 'PyBot', read_only = False,preprocessors=['chatterbot.preprocessors.clean_whitespace','chatterbot.preprocessors.convert_to_ascii','chatterbot.preprocessors.unescape_html'], logic_adapters = ['chatterbot.logic.MathematicalEvaluation','chatterbot.logic.BestMatch'])
ind = 1
if st.sidebar.button('Initialize bot'):
    trainer2 = ListTrainer(bot) 
    trainer2.train(train)
    st.title("Your bot is ready to talk to you")
    ind = ind +1   
    
user_input = get_text()
if user_input:
    st.text_area("Bot:", value=bot.get_response(user_input), height=200, max_chars=None, key=None)
else:
    st.text_area("Bot:", placeholder="I am ready to Talk with you", height=200, max_chars=None, key=None)