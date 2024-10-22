import os
from dotenv import load_dotenv 
load_dotenv()
db_user = os.getenv("db_user")
db_password = os.getenv("db_password")
db_host = os.getenv("db_host")
db_name = os.getenv("db_name")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_2")
print(OPENAI_API_KEY)


from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.memory import ChatMessageHistory

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from table_details import table_chain as select_table
from prompts import final_prompt, answer_prompt
import streamlit as st
import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_csv_agent
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import pandas as pd

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY_2')
@st.cache_resource
def get_chain():
    """
    Creates a conversation chain for interacting with CSV data.
    
    Args:
        csv_file_path (str): Path to the CSV file
        openai_api_key (str): OpenAI API key
        
    Returns:
        chain: A LangChain chain for conversing about the CSV data
    """
    # Initialize the LLM
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4",
        api_key=OPENAI_API_KEY
    )
    
    try:
        # Read and process the CSV
        df = pd.read_csv('student_database.csv')
        csv_data = df.to_string()
        
        # Split the text for vectorization
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_text(csv_data)
        
        # Create vector store
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_texts(splits, embeddings)
        retriever = vectorstore.as_retriever()
        
        # Define the prompt template without chat history
        template = """
        You are an AI assistant specialized in analyzing CSV data. Use the following context
        to provide accurate and helpful responses about the data.

        Context: {context}
        Human Question: {question}

        Please provide a clear, direct answer based on the data. If you're making calculations or 
        observations, explain your reasoning. If you cannot answer based on the available data, 
        say so clearly. If you are provided SQL query try finding what is required in SQL data in the CSV file.

        Answer:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create the chain with proper input handling
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        chain = (
            {
                "context": lambda x: format_docs(retriever.get_relevant_documents(x["question"])),
                "question": lambda x: x["question"]
            }
            | prompt
            | llm
        )
        
        return chain
    
    except Exception as e:
        print(f"Error initializing chain: {str(e)}")
        raise

# @st.cache_resource
# def get_chain():
#     print("Creating chain")
#     db = SQLDatabase.from_uri(
#     f"postgresql://{db_user}:{db_password}@{db_host}/{db_name}")    
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=)
#     generate_query = create_sql_query_chain(llm, db, final_prompt)
#     execute_query = QuerySQLDataBaseTool(db=db)
#     rephrase_answer = answer_prompt | llm | StrOutputParser()
#     # chain = generate_query | execute_query
#     chain = (
#             RunnablePassthrough.assign(table_names_to_use=select_table) |
#             RunnablePassthrough.assign(query=generate_query).assign(
#                 result=itemgetter("query") | execute_query
#             )
#             | rephrase_answer
#     )

#     return chain


def create_history(messages):
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history


def invoke_chain(question, messages):
    chain = get_chain()
    history = create_history(messages)
    response = chain.invoke({"question": question, "top_k": 3, "messages": history.messages})
    
    error_keywords = [
        "error in the SQL query",
        "type mismatch",
        "cast",
        "column is likely of type",
    ]
    
    # if any(keyword in response.lower() for keyword in error_keywords):
    #     response = "I'm sorry, I couldn't understand the question or find the relevant information."

    history.add_user_message(question)
    history.add_ai_message(response.content)
    return response.content
