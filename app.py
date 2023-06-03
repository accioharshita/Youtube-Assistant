import streamlit as st
from langchain_main import creating_db, get_response
import textwrap




#setting up the title
st.title("Hello, I'm `Ray` your Youtube Assistant 👓  ")

#User input video
video_url= st.text_input('Please enter your Youtube link here!')

#User input question
query= st.text_input('Please enter your question here 👇')


def answer():
    db= creating_db(video_url)
    response, docs = get_response(db, query, k=5)

    if video_url and query:
        st.write(textwrap.fill(response, width=50))
    


#aesthetics
st.button('Find the Answer', on_click=answer)