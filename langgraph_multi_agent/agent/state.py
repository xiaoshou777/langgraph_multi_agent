from typing import TypedDict, Annotated
from operator import add
from langchain_core.messages import AnyMessage

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    type: str