import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub


if __name__ == "__main__":
    pdf_fp = "2210.03629v3.pdf"
    loader=PyPDFLoader(file_path=pdf_fp)
    documents= loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index")
    
    new_vectorstore = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    
    retieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(OpenAI(), retieval_qa_chat_prompt)
    retieval_chain = create_retrieval_chain(new_vectorstore.as_retriever(), combine_docs_chain)
    
    res = retieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    print(res["answer"])