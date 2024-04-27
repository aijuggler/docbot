from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langsmith import traceable
import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from typing import Tuple

# Define your variables
variables = {
    "LANGCHAIN_TRACING_V2": "True",
    "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
    "LANGCHAIN_API_KEY": "ls__e70c98c314364c01999d3b06f1d6cea6",
    "LANGCHAIN_PROJECT": "docbot_llama3",
    "GROQ_API_KEY": "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"
}

# Set each variable as an environment variable
for key, value in variables.items():
    os.environ[key] = value

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
    pages = loader.load_and_split(text_splitter=text_splitter)
    return pages

def create_embeddings(docs: PyPDFLoader, pdf_name):
    embeddings = HuggingFaceEmbeddings()
    vec_check = os.listdir('doc_embedding_store')
    if pdf_name in vec_check:
        print(f'Embedding for document {pdf_name} already present in storage')
        print(f'*******-----Using Saved Embeddings-------******')
        vectorStore = FAISS.load_local(f'doc_embedding_store/{pdf_name}', embeddings, 
                                       allow_dangerous_deserialization=True)
    else:
        print(f'*******-----Creating Embeddings-------******')
        print(f'Embedding for document {pdf_name} not created in storage')
        vectorStore = FAISS.from_documents(docs, embeddings)
        vectorStore.save_local(f'doc_embedding_store/{pdf_name}')
    return vectorStore


def query_llm(vector_store: FAISS)-> Tuple[ConversationalRetrievalChain, ConversationBufferMemory]:
    retriever = vector_store.as_retriever()
    chat = ChatGroq(temperature=0, model_name="llama3-8b-8192")
    memory = ConversationBufferMemory(
    memory_key='chat_history', return_messages=True, output_key='answer')
    conversation_chain = (ConversationalRetrievalChain.from_llm
                        (llm=chat,
                        retriever=retriever,
                        # condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                        memory=memory,
                        return_source_documents=True,verbose=False))
    print("Conversational Chain created for the LLM using the vector store")
    return conversation_chain,memory

def indexing(pdf,pdf_name):
    # pdf_name = pdf.split("\\")[1].split(".")[0]
    document = load_pdf(pdf_path=pdf)
    vectorStores = create_embeddings(docs=document, pdf_name=pdf_name)
    return vectorStores













    


