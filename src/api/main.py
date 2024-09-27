import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import langchain.globals

from src.chains import preprocess_pdf
from src.voice.websocket import websocket_endpoint
from src.utils import convert_response_output


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")


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
    # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_methods=["*"],
    # Allows all headers (e.g., Authorization, Content-Type, etc.)
    allow_headers=["*"],
)

UPLOAD_DIRECTORY = "uploads"
Path(UPLOAD_DIRECTORY).mkdir(parents=True, exist_ok=True)
SESSION_ID = os.environ.get("SESSION_ID", "test")

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


# @app.post("/init-copilot")
# def init_copilot(req: UserInput):
#     response = init_mars_agent(session_id=req.user_id)
#     return convert_response_output(response)


@app.post("/ask-copilot")
def ask_copilot(req: UserInput):
    from src.agent import get_mars_agent

    mars_agent = get_mars_agent()
    response = mars_agent.invoke({
        "input": str(req)
    },
        config={"configurable": {"session_id": SESSION_ID}},
    )
    print(f"Agent: {response['output']}")
    return convert_response_output(response)
