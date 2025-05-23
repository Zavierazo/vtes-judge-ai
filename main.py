import os
import requests
import logging
import sys
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# from dotenv import load_dotenv
# load_dotenv()

class CardNameInput(BaseModel):
    song: str = Field(
        description="exact card name from csv Name column")

# Initialize ruling tool
@tool("ruling_by_name", return_direct=True, args_schema=CardNameInput)
def ruling_by_name(name: str) -> list:
    """Extract the ruling from card name."""
    url = f"https://api.krcg.org/card/{name}"
    response = requests.get(url)
    if response.status_code==200:
        response = response.json()
        return response["rulings"] if "rulings" in response else []
    else:
        return []

app = FastAPI()
LOG = logging.getLogger('uvicorn.error')


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

#store = LocalFileStore("./.cache/")

#embedding = CacheBackedEmbeddings.from_bytes_store(
#    openai_embedding, store, namespace=openai_embedding.model
#)

# Initialize rulebook vector database
LOG.info("Loading rulebook...")
pdf_path = "./data/rulebook.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

rulebook_vectordb = Chroma.from_documents(texts, embedding, persist_directory="./.chroma/rulebook")

rulebook_retriever = rulebook_vectordb.as_retriever()

#Initialize csv vector database
LOG.info("Loading csv...")
loader = DirectoryLoader('./data/csv', glob='*.csv', loader_cls=CSVLoader, loader_kwargs={'encoding': 'utf-8'})
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

csv_vectordb = Chroma.from_documents(texts, embedding, persist_directory="./.chroma/csv")

csv_retriever = csv_vectordb.as_retriever()

#Initialize tournament rules vector database
LOG.info("Loading tournament rules...")
loader = WebBaseLoader(["https://www.vekn.net/tournament-rules", "https://www.vekn.net/judges-guide"])

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

tournament_rules_vectordb = Chroma.from_documents(texts, embedding, persist_directory="./.chroma/tournament_rules")

tournament_rules_retriever = tournament_rules_vectordb.as_retriever()

#Initialize general rules vector database
LOG.info("Loading general rules...")
loader = WebBaseLoader("https://www.vekn.net/general-rulings")

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

general_rules_vectordb = Chroma.from_documents(texts, embedding, persist_directory="./.chroma/general_rules")

general_rules_retriever = general_rules_vectordb.as_retriever()

#Initialize general rules vector database
LOG.info("Loading imbued rules...")
loader = WebBaseLoader("https://www.vekn.net/rulebook/appendix-imbued-rules")

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

imbued_rules_vectordb = Chroma.from_documents(texts, embedding, persist_directory="./.chroma/imbued_rules")

imbued_rules_retriever = imbued_rules_vectordb.as_retriever()

# Initialize ChatOpenAI
LOG.info("Initializing ChatOpenAI...")
llm = ChatOpenAI(
    model_name='gpt-4o',
    temperature=0.0
)

# Create tools for the agent
LOG.info("Creating tools...")
tools = [
    Tool(
        name="CSV",
        func=csv_retriever.get_relevant_documents,
        description="Useful for retrieving relevant documents with info about crypt and library texts and disciplines",
    ),
    Tool(
        name="Ruling",
        func=ruling_by_name,
        description="Useful for extracting the ruling from card name. Card name should be exact match from csv Name column",
    ),
    Tool(
        name="Tournament",
        func=tournament_rules_retriever.get_relevant_documents,
        description="Useful for retrieving relevant documents with info about tournament/event/judge rulings",
    ),
    Tool(
        name="General",
        func=general_rules_retriever.get_relevant_documents,
        description="Useful for retrieving relevant documents with info about general rulings that apply to all cards",
    ),
    Tool(
        name="Imbued",
        func=imbued_rules_retriever.get_relevant_documents,
        description="Useful for retrieving relevant documents with info about imbued, conviction, power rulings",
    )
]

# Creating Prompt
LOG.info("Creating prompt...")
prompt = ChatPromptTemplate.from_messages([
    ("system", 
        "You are a helpful AI assistant for Vampire: The Eternal Struggle (VTES) trading card game. "
        "Address questions exclusively related to VTES. "
        "Use the provided tools to answer the user's question. "
        "Use the CSV tool to extract the card name & text. "
        "Use the General tool to extract the general rulings that apply to all cards. "
        "Use the Ruling tool to extract the ruling from card name using the exact card name from CSV. "        
        "Use the Tournament tool to extract the tournament/event/judge rulings. "
        "Use the Imbued tool to extract the imbued, conviction, power rulings. "
        "Add source of the answer in the end of your answer. "
        "If source is rulebook add page number with url https://www.blackchantry.com/utilities/rulebook/. "
        "If you don't know the answer, just say that you don't know, don't try to make up an answer "
        "If you don't find the card in the CSV tool, say that you didn't find the card with the name. "
        "If card have ruling, add all rulings in the end of your answer with links to the ruling. "
        "If card have no ruling, add 'No ruling found' in the end of your answer. "
        "If question is about a card and you know the id of the card, add the card image at the end with path /assets/img/cards/$cardId.jpg where $cardId is the id of the card. "
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
    ("human", "Relevant information: {context}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Agent
LOG.info("Creating agent...")
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Create API model
class ChatMessage(BaseModel):
    type: str 
    content: str
class Query(BaseModel):
    chat_history: list[ChatMessage]
    question: str

# Create API Endpoint
LOG.info("Initializing API endpoint...")
def get_message_object(message):
    if message.type == "human":
        return HumanMessage(content=message.content)
    elif message.type == "ai":
        return AIMessage(content=message.content)
    else:
        # Handle any unexpected message type as needed
        raise ValueError(f"Unexpected message type: {message.type}")

@app.post("/ask")
async def ask_question(query: Query):
    LOG.info(f"Query: {query.question}")
    response = agent_executor.invoke({
        "context": rulebook_retriever, 
        "question": query.question,
        "chat_history": [get_message_object(message) for message in query.chat_history]
    })
    return {"answer": response['output']}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
