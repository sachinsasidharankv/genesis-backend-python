from langchain.prompts import PromptTemplate
from langchain.agents import tool
from langchain_core.messages import SystemMessage
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser

from src.models import ActionModel, Action
from src.utils import get_llm


@tool
def action_tool(student_query: str) -> str:
    """Tool used to identify which action to take based on student query"""
    llm = get_llm(temperature=0)

    action_prompt_template = """
    You are tasked with identifying which action to take from a list of actions based on student query.

    Student query: {student_query}
    Actions: {actions}
    """

    action_prompt = PromptTemplate(
        input_variables=["student_query", "actions"],
        template=action_prompt_template
    )

    @chain
    def select_action(inputs: dict) -> str | list[str] | dict:
        response = llm.invoke([
            SystemMessage(
                content=[
                    {"type": "text", "text": inputs["prompt"]},
                ]
            )
        ])

        return response.content

    parser = JsonOutputParser(pydantic_object=ActionModel)

    action_chain = select_action | parser

    return action_chain.invoke({
        "prompt": action_prompt.format_prompt(
           actions=', '.join([action.value for action in Action]),
           student_query=student_query
        ).to_string()
    })
