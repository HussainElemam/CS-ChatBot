# from langchain_community.llms.ollama import Ollama
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from src.create_database import VECTORSTORE_PATH, MODEL, EMBEDDING_MODEL

PROMPT_TEMPLATE = """
Answer the question based on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text, history):
    # Prepare the DB.
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)
    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt_value = prompt_template.format_messages(context=context_text, question=query_text)

    print("final prompt value:", prompt_value)

    user_message = prompt_value[0]

    chat_history = get_history(history)
    chat_history.append(user_message)

    model = OllamaLLM(model=MODEL)
    response_text = model.invoke(chat_history)

    return response_text

def get_history(raw_history):
    history = []
    for chat in raw_history:
        if chat['role'] == 'user':
            history.append(HumanMessage(content=chat['content']))
        else:
            history.append(AIMessage(content=chat['content']))

    return history
