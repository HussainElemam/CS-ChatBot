# comprehensive tutorial: https://python.langchain.com/docs/tutorials/rag/

import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
import shutil

MODEL = "qwen2.5:3b"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/course_materials")
VECTORSTORE_PATH = os.path.join(BASE_DIR,  "../data/vector_store")

def main():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_vectorstore(chunks)


def load_documents():
    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".pptx": UnstructuredPowerPointLoader,
    }

    all_documents = []
    for ext, loader_cls in loaders.items():
        loader = DirectoryLoader(DATA_PATH, glob=f"**/*{ext}", loader_cls=loader_cls)
        documents = loader.load()
        all_documents.extend(documents)


    print(f'loaded docs {len(all_documents)} documents')
    return all_documents


def split_text(docs: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
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