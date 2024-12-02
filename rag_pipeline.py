from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
import os
import tempfile
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

EMBEDDINGS_MODEL = "nomic-embed-text"  # Model changed to 'nomic-embed-text'
LLM_MODEL="llama3.2:3b"


def load_and_split_pdfs(pdf_files):
    documents = []
    with tempfile.TemporaryDirectory() as temp_dir:  # Create a temporary directory
        for file in pdf_files:
            temp_file_path = os.path.join(temp_dir, file.name)
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file.getbuffer())  # Save the BytesIO content to the temp file

            loader = PyPDFLoader(temp_file_path)
            documents.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

def create_vectorstore(chunks):
    #embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Using the new method for Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",  # Specify your Chroma DB persistence directory
        collection_name='local_rag_db'  # Specify the collection name
    )
    return vectorstore

def initialize_qa_chain(vectorstore):
  #llm = OllamaLLM(model=LLM_MODEL)
  llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
  retriever = vectorstore.as_retriever()
  memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
  
  # No need to pass "question" as an argument
  return ConversationalRetrievalChain.from_llm(
      llm=llm,
      retriever=retriever,
      memory=memory
  )
