import streamlit as st
from rag_pipeline import load_and_split_pdfs, create_vectorstore, initialize_qa_chain

st.title("Simple RAG Pipeline with Chat History")

# File upload section
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

# Initialize session state for QA chain
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if uploaded_files:
    st.write("Processing files...")
    chunks = load_and_split_pdfs(uploaded_files)
    vectorstore = create_vectorstore(chunks)
    st.session_state.qa_chain = initialize_qa_chain(vectorstore)
    st.write("Processing complete! Ready to answer queries.")

# Chat interface
if st.session_state.qa_chain:
    query = st.text_input("Ask a question about the documents:")
    
    if query:
        # Pass the correct input key
        result = st.session_state.qa_chain({"question": query})  
        answer = result["answer"]  # Use the correct output key
        chat_history = result["chat_history"]
        
        # Display answer
        st.write("Answer:", answer)
        
        # Display chat history
        st.write("Chat History:")
        for message in chat_history:
            if hasattr(message, 'role') and hasattr(message, 'content'):
                st.write(f"**{message.role}:** {message.content}")
            else:
                # Handle the case where attributes are missing (e.g., print a warning)
                st.write("Error: Unexpected message format in chat history.")

