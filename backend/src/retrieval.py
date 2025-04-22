from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
# from rag import load_vectorstore


def retrieve_data(query):
    embeddings_model = OllamaEmbeddings(model="llama3")
    embedded_query = embeddings_model.embed_query(query)


def load_vectorstore():
    embeddings = OllamaEmbeddings(model="mistral")
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)


def get_chain():
    llm = Ollama(model="mistral")
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain

def get_response(query):
    chain = get_chain()
    return chain.run(query)