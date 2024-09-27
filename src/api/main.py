import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.chains import preprocess_pdf
from src.voice.websocket import websocket_endpoint
import langchain.globals

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv(ROOT_DIR / ".env")


print(os.getenv("OPENAI_API_KEY"))

is_debug = os.environ.get("DEBUG_MODE") == "true"
langchain.globals.set_debug(is_debug)
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
SUBTOPICS_DIR = os.environ.get("SUBTOPICS_DIR", "subtopics")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SUBTOPICS_DIR, exist_ok=True)

class UserInput(BaseModel):
    context: str
    query: str



app = FastAPI(title="FastAPI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,  # Allows sending cookies and credentials
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers (e.g., Authorization, Content-Type, etc.)
)

UPLOAD_DIRECTORY = "uploads"
Path(UPLOAD_DIRECTORY).mkdir(parents=True, exist_ok=True)

app.add_api_websocket_route("/ws", websocket_endpoint)


@app.get("/")
def hello_world():
    return {"Hello": "World"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...), user_input: str = Form(...)):
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    print(user_input)
    print(preprocess_pdf(filepath=file_location, index_name=file.filename))
    return {"preprocessed": file_location}


# @app.get("/generate-question-paper")
# def generate_question_paper(req: UserInput):
#     qp_generation(req.user_input)
#     return {"question_paper": req}


@app.get("/feedback")
def feedback(req: UserInput):
    from src.agent import get_our_agent
    from langchain.memory import ConversationBufferMemory

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    agent_executor = get_our_agent(memory=memory)

    chat_history = memory.buffer_as_messages
    response = agent_executor.invoke({
        "input": f"Context: {req.context}\nStudent query: {req.query}",
        "chat_history": chat_history,
    })
    print(response["output"])


@app.post("/agent")
def agent(req: UserInput):
    return {
        "context": req.context,
        "query": req.query
    }

@app.post("/ask-copilot")
def ask_copilot(req: UserInput):
    from src.agent import get_our_agent
    from langchain.memory import ConversationBufferMemory

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    agent_executor = get_our_agent(memory=memory)

    chat_history = memory.buffer_as_messages
    response = agent_executor.invoke({
            "input": str(req),
            "chat_history": chat_history,
        })
    print(f"Agent: {response['output']}")
    return {"response": response['output']}
