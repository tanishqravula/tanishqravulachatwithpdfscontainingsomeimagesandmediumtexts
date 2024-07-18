import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pytesseract import Output, TesseractError
from functions import convert_pdf_to_txt_pages, convert_pdf_to_txt_file, save_pages, displayPDF, images_to_txt
import fitz  # PyMuPDF
import pdf2image
from pdf2image.exceptions import PDFPageCountError
import asyncio

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            texts, nbPages = images_to_txt(pdf.read(), 'eng')
            text_data_f = "\n\n".join(texts)
            text += text_data_f
        except PDFPageCountError as e:
            st.write(f".")
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, making sure to provide all the details. Elaborate on every point
    and include examples, explanations, and any relevant information. If the answer is not in the provided context, just say, "answer is not available in the context."
    Do not provide incorrect information.\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=1.0)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask Questions from the PDF Files uploaded .. ✍️📝 like example of prompt-explain (concept or topic) as per the context briefly in N(N=500 or 100 or 200 or 400 or 300 or 1000 etc..) words or prompt-explain the concept of (topic or concept name) as per the context briefly in N(N= number) words or write a full fledged code based on this information given or Explain Chapter 2 Introduction to the Relational Model as per the context in 11000-16000 words.plzz don't involve any outside knowledge or outside information plzz include all the context u write from this pdf only plzz.i have exam on this topic if I write outside content that is not there in book I will get 0 marks.so plzzz help me"}]

async def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        if not docs:
            return "No relevant documents found."

        chain = get_conversational_chain()

        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        if not response or 'output_text' not in response:
            return "No valid response generated."

        return response['output_text']

    except Exception as e:
        st.error(f"Error occurred: {e}")
        return None

def main():
    st.set_page_config("Tanishq Ravula PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖 ")

    with st.sidebar:
        st.image("Robot.png")
        st.write("---")
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing completed successfully!")
        
        st.write("---")
        st.write("Tanishq Ravulas AI PDF Chatbot")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask Questions from the PDF Files uploaded .. ✍️📝 like example of prompt-explain (concept or topic) as per the context briefly in N(N=500 or 100 or 200 or 400 or 300 or 1000 etc..) words or prompt-explain the concept of (topic or concept name) as per the context briefly in N(N= number) words or write a full fledged code based on this information given or Explain Chapter 3: Introduction to SQL as per the context in 11000-16000 words.plzz don't involve any outside knowledge or outside information plzz include all the context u write from this pdf only plzz.i have exam on this topic if I write outside content that is not there in book I will get 0 marks.so plzzz help me"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(user_input(prompt))
                if response:
                    st.write(response)
                    message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(message)
                else:
                    st.write("No valid response generated.")

if __name__ == "__main__":
    main()
