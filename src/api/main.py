import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

from src.chains import preprocess_pdf, qp_generation
from src.voice.websocket import websocket_endpoint


class UserInput(BaseModel):
    user_input: str


class Feedback(BaseModel):
    context: dict
    user_input: str


app = FastAPI(title="FastAPI")

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


@app.get("/generate-question-paper")
def generate_question_paper(req: UserInput):
    qp_generation(req.user_input)
    return {"question_paper": req}


@app.get("/feedback")
def feedback(req: Feedback):
    from src.agent import get_our_agent
    from langchain.memory import ConversationBufferMemory

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    agent_executor = get_our_agent(memory=memory)

    chat_history = memory.buffer_as_messages
    response = agent_executor.invoke({
            "input": f"Context: {req.context}\nStudent query: {req.user_input}",
            "chat_history": chat_history,
        })
    print(response["output"])
