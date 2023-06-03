import openai
import os
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap

#importing the .env file containing the api key
load_dotenv(find_dotenv())
embeddings= OpenAIEmbeddings()


#establishing the key
api_key = os.environ['OPENAI_API_KEY']


#creating a database
def creating_db(video_url):

    loader= YoutubeLoader.from_youtube_url(video_url)
    transcript= loader.load()

    #to breakdown the enormous amount of tokens we will get from the transcript as we have a limited set we can input
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    #this is just a list with the bunch of splits from the above
    docs= text_splitter.split_documents(transcript)
    
    #the final database
    '''
    when a user asks a question, this database will be used to perform the similarity search and 
    generate output based on that 
    '''

    db= FAISS.from_documents(docs, embeddings) #embeddings are the vectors we convert the text over into
    return db



#creating another function to get response from querying the above database
def get_response(db, query, k=5):

    '''
    gpt can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    '''

    docs= db.similarity_search(query, k=k)

    #joining them into one single string
    docs_page_content = " ".join([d.page_content for d in docs])

    chat= ChatOpenAI(temperature=0.4)


    #template for the system message prompt

    template= '''
              You are a helpful assistant who can answer question from Youtube videos based on the video's transcript: {docs}

              Only use the factual information from transcript to answer the question.

              If you feel like you don't have enough information to answer the question, say: "Sorry, I cannot answer that".

              Your answer should be verbose and detailed.

              '''
    
    system_message_prompt= SystemMessagePromptTemplate.from_template(template)

    #Human question prompt

    human_template= 'Answer the following question: {question}'

    human_message_prompt= HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt= ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]

    )


    #chaining

    chain= LLMChain(llm=chat, prompt=chat_prompt)

    response= chain.run(question=query, docs= docs_page_content)
    response = response.replace("\n", "")

    return response, docs




#calling the functions:

# video_url= input('Please enter the url: ')

# db= creating_db(video_url)

# query= input('Please enter your question: ')
# response, docs = get_response(db, query, k=5)

# print(textwrap.fill(response, width=50))







      





