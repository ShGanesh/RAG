import os
import textwrap
import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai

# Configuring Gemini API
genai.configure(api_key='AIzaSyClRjxewRDN8gqhwkMzCMyUutGBVZEyI8g')

emb_model = 'models/text-embedding-004'
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Define file paths
pdf_folder = "./pdfs"
embedding_file = "./embeddings.pkl"

os.makedirs(pdf_folder, exist_ok=True)

def embed_fn(title, text):
    return genai.embed_content(model=emb_model, content=text, task_type="retrieval_document", title=title)["embedding"]

def save_embeddings(df):
    df.to_pickle(embedding_file)

def load_embeddings():
    if os.path.exists(embedding_file):
        return pd.read_pickle(embedding_file)
    else:
        return None

def find_best_passage(query, dataframe):
    query_embedding = genai.embed_content(model=emb_model, content=query, task_type="retrieval_query")["embedding"]
    dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding)
    idx = np.argmax(dot_products)
    return dataframe.iloc[idx]['Content']

def make_prompt(query, relevant_passage):
    prompt = textwrap.dedent("""
        As a helpful HR Assistant, follow the given instructions: 
        **Instructions:**
        1. Review the provided extract of the policy document in the context of the question.
        2. If the passage is relevant to answering the question, use it to formulate your response.
        3. If the passage is not relevant or doesn't provide useful information for answering the question, you may ignore it.
        
        **Question:**
        {query}
        
        **Passage:**
        {relevant_passage}
        
        **Answer:**
        """).format(query=query, relevant_passage=relevant_passage)
    return prompt

def main():
    st.title("Company Policy Document Assistant")

    # Display existing PDFs
    st.subheader("Existing PDFs:")

    # File uploader
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    # Button to recreate embeddings
    if st.button("Recreate Embeddings"):
        st.write("Recreating embeddings...")
        texts = []
        for uploaded_file in uploaded_files:
            pdf_path = os.path.join(pdf_folder, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            reader = PdfReader(pdf_path)
            text = ''
            for i in range(len(reader.pages)):
                text += ' ' + reader.pages[i].extract_text()

            text = text.replace('\n', '')
            texts.append({'Title': uploaded_file.name, 'Content': text})

        df = pd.DataFrame(texts)
        df.columns = ['Title', 'Content']
        df['Embeddings'] = df.apply(lambda row: embed_fn(row['Title'], row['Content']), axis=1)
        save_embeddings(df)
        st.write("Embeddings recreated and saved.")
    
    # Display and handle existing embeddings
    df = load_embeddings()
    if df is not None:
        st.write("Embeddings loaded.")
    else:
        st.write("No embeddings found. Please upload PDFs and recreate embeddings.")
        df = pd.DataFrame(columns=['Title', 'Content', 'Embeddings'])

    # Query input form
    with st.form(key='query_form'):
        query = st.text_input("Enter your query")
        submit_button = st.form_submit_button(label='Submit')

    if submit_button and query and df is not None and not df.empty:
        # Find and display extracted passage
        st.subheader("Extracted Passage:")
        answer = find_best_passage(query, df)
        st.write(answer)

        # Generate and display final answer
        st.subheader("Final Answer:")
        prompt = make_prompt(query, answer)
        final_answer = model.generate_content(prompt)
        st.write(final_answer.text)

if __name__ == "__main__":
    main()
