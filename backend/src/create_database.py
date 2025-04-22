# comprehensive tutorial: https://python.langchain.com/docs/tutorials/rag/

import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS, Chroma
import shutil

MODEL = "qwen2.5:0.5b"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/course_materials")
VECTORSTORE_PATH = os.path.join(BASE_DIR,  "../data/vector_store")

def main():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_vectorstore(chunks)


# for now, it will only load pdf files
# I will later implement other types
# see: https://python.langchain.com/docs/how_to/#document-loaders
# and: https://python.langchain.com/docs/integrations/document_loaders/
def load_documents():
    # loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    loader = PyPDFDirectoryLoader(DATA_PATH, glob="*.pdf")
    docs = loader.load()

    print(f'loaded docs {len(docs)}')
    return docs


def split_text(docs: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)

    print(f"Split documents {len(all_splits)} sub-documents.")

    example = all_splits[0]
    print(example.page_content)
    print(example.metadata)

    return all_splits


# for embedding see: https://python.langchain.com/docs/how_to/embed_text/
# vector store stuff: https://python.langchain.com/docs/integrations/vectorstores/faiss/
def save_to_vectorstore(chunks: list[Document]):
    if os.path.exists(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH)

    embeddings = OllamaEmbeddings(model=MODEL)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=VECTORSTORE_PATH)
    print(f'Saved to {VECTORSTORE_PATH}')


def load_vectorstore():
    embeddings = OllamaEmbeddings(model=MODEL)
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)



if __name__ == "__main__":
    main()