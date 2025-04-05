from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langsmith import Client
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

load_dotenv()

#print(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# chat model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    max_length=128,
    temperature=0.1,
    task="text-generation",  #HuggingFaceEndpoint Requires Explicit task Argument
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# vector store
vector_store = InMemoryVectorStore(embeddings)

vector_store_ready = False

def vector_store_exists():
    return vector_store_ready

def build_vector_store():
    # Load and chunk contents of the blog
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # Index chunks
    _ = vector_store.add_documents(documents=all_splits)

    vector_store_ready = True

def answer_query(query):
    # Define prompt for question-answering
    #client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
    #prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)
    #docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    #messages = prompt.invoke({"question": state["question"], "context": docs_content})
    template = """Question: {question}

    Answer: Let's think step by step."""

    # Define prompt for question-answering
    prompt = PromptTemplate.from_template(template)      

    # Define application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        llm_chain = prompt | llm
        response = llm_chain.invoke(state["question"])
        return {"answer": response}

    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    print("query", query)
    response = graph.invoke({"question": query})
    print(response["answer"])
    result = {}
    result["answer"] = response["answer"]
    result["sources"] = 'LLMBot'

    return result