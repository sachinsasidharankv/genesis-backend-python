from enum import Enum
from pydantic import BaseModel


class Chain(str, Enum):
    FEEDBACK = "FEEDBACK"
    GUIDANCE = "GUIDANCE"
    TEACHER = "TEACHER"
    AGENT = "AGENT"


class UserInput(BaseModel):
    context: dict
    query: str
    chain: Chain
