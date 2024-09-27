from pydantic import BaseModel

class UserInput(BaseModel):
    context: dict
    query: str
