import json
from langchain.prompts import PromptTemplate
from langchain.agents import tool
from langchain_core.messages import SystemMessage
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser

from src.models import ActionModel, Action
from src.utils import get_llm


@tool(return_direct=True)
def action_tool(student_query: str) -> str:
    """Tool used when the student has raised a concern or query"""
    llm = get_llm(temperature=0)

    action_prompt_template = """
    You are tasked with providing the student with a list of actions along with explaining why each action helps the student.
    Do not mention the action as such in the question because it is an enum, so use natural language always.
    Start by giving all the actions and only if the student has asked otherwise, give specific actions.

    Student query: {student_query}
    Actions: {actions}
    """

    action_prompt = PromptTemplate(
        input_variables=["student_query", "actions"],
        template=action_prompt_template
    )

    parser = JsonOutputParser(pydantic_object=ActionModel)

    @chain
    def select_action(inputs: dict) -> str | list[str] | dict:
        response = llm.invoke([
            SystemMessage(
                content=[
                    {"type": "text", "text": inputs["prompt"]},
                    {"type": "text", "text": parser.get_format_instructions()},
                ]
            )
        ])

        return response.content

    action_chain = select_action | parser

    return json.dumps(
        action_chain.invoke({
            "prompt": action_prompt.format_prompt(
                actions=', '.join([action.value for action in Action]),
                student_query=student_query
            ).to_string()
        })
    )
