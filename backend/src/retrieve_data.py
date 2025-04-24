from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from src.create_database import VECTORSTORE_PATH, MODEL

PROMPT_TEMPLATE = """
Answer the question based on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
GETTING_CONTEXT_PROMPT = """
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""

def query_rag(query_text, history):
    model = Ollama(model=MODEL)

    # summarize the conversation to get the context
    prompt_template = ChatPromptTemplate.from_template(GETTING_CONTEXT_PROMPT)
    summery_prompt_value = prompt_template.format_messages(chat_history=history, question=query_text)
    summarized_conversation = model.invoke(summery_prompt_value)

    print("\n\nsummarized conversation:\n", summarized_conversation, "\n\n\n")

    # Prepare the DB.
    embeddings = OllamaEmbeddings(model=MODEL)
    db = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(summarized_conversation, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print("context text:\n", context_text)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt_value = prompt_template.format_messages(context=context_text, question=query_text)

    print("final prompt value:", prompt_value)

    user_message = prompt_value[0]

    chat_history = get_history(history)
    chat_history.append(user_message)

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
