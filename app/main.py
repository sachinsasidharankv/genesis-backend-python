from fastapi import FastAPI

app = FastAPI(title="FastAPI")


@app.get("/")
def hello_world():
    return {"Hello": "World"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict():
    return {"prediction": "fake news"}
