examples = [
    {
        "input": "List all students graduating in 2025.",
        "query": "SELECT * FROM student_detils WHERE graduation_year = 2025;"
    },
    {
        "input": "Get the names of students with no backlogs.",
        "query": "SELECT name FROM student_detils WHERE backlogs = 0;"
    },
    {
        "input": "Show details of students studying Physics.",
        "query": "SELECT * FROM student_detils WHERE subjects LIKE '%Physics%';"
    },
    {
        "input": "Retrieve the names of students who got an A grade.",
        "query": "SELECT name FROM student_detils WHERE grades LIKE '%A%';"
    },
    {
        "input": "List all students with a class ranking better than 50.",
        "query": "SELECT name, class_ranking FROM student_detils WHERE class_ranking < 50;"
    },
    {
        "input": "What is the phone number of the student named 'John Smith'?",
        "query": "SELECT phone_number FROM student_detils WHERE name = 'John Smith';"
    },
    {
        "input": "Count the number of students graduating in each year.",
        "query": "SELECT graduation_year, COUNT(*) as student_count FROM student_detils GROUP BY graduation_year;"
    },
    {
        "input": "Find the average number of backlogs across all students.",
        "query": "SELECT AVG(backlogs) as average_backlogs FROM student_detils;"
    },
    {
        "input": "List the names of top 10 students based on class ranking.",
        "query": "SELECT name, class_ranking FROM student_detils ORDER BY class_ranking ASC LIMIT 10;"
    },
    {
        "input": "Show the names of students studying both Mathematics and Computer Science.",
        "query": "SELECT name FROM student_detils WHERE subjects LIKE '%Mathematics%' AND subjects LIKE '%Computer Science%';"
    }
]


from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
import streamlit as st
import os 
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def get_example_selector():
    try:
        embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY_2'))
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            embeddings,
            FAISS,
            k=2,
            input_keys=["input"],
        )
        return example_selector
    except Exception as e:
        st.error(f"Error creating example selector: {str(e)}")
        raise