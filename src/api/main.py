import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import langchain.globals

from src.api.models import UserInput, Chain, Context
from src.chains import (
    run_feedback_chain,
    run_guidance_chain,
    run_teacher_chain
)
from src.voice.websocket import websocket_endpoint, language_model_processor
from src.utils import convert_response_output


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")


is_debug = os.environ.get("DEBUG_MODE") == "true"
langchain.globals.set_debug(is_debug)
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
SUBTOPICS_DIR = os.environ.get("SUBTOPICS_DIR", "subtopics")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SUBTOPICS_DIR, exist_ok=True)


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

app.add_api_websocket_route("/ws", websocket_endpoint)


@app.get("/")
def hello_world():
    return {"Hello": "World"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask-copilot")
def ask_copilot(req: UserInput):
    from src.agent import get_mars_agent

    mars_agent = get_mars_agent()
    session_id = os.environ.get("SESSION_ID", "test")
    response = mars_agent.invoke({
        "input": str(req)
    },
        config={"configurable": {"session_id": session_id}},
    )
    print(f"Agent: {response['output']}")
    return convert_response_output(response['output'])


@app.post("/ask-chain")
def get_feedback(req: UserInput):
    if req.chain == Chain.FEEDBACK:
        response = run_feedback_chain(
            exam_results_dict=req.context.get("exam_results_dict"),
            student_summary=req.context.get("student_summary")
        )
    elif req.chain == Chain.GUIDANCE:
        response = run_guidance_chain(
            student_query=req.query,
            question_dict=req.context.get("question_dict"),
            student_summary=req.context.get("student_summary")
        )
    elif req.chain == Chain.TEACHER:
        response = run_teacher_chain(
            student_query=req.query,
            highlighted_text=req.context.get("highlighted_text"),
            reference_page_base64=req.context.get("reference_page_base64")
        )
    else:
        response = None

    print(f"Chain: {response}")
    return convert_response_output(response)

@app.post("/set-context")
def set_context(req: Context):
    language_model_processor.memory.chat_memory.clear()
    language_model_processor.memory.chat_memory.add_user_message(req.context)
    return {"status": "ok"}
