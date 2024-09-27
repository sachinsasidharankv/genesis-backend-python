from langchain.prompts import PromptTemplate
from langchain.agents import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import chain

from src.utils import get_llm


@tool
def reasoning_tool(
    student_query: str
) -> str:
    "Reasoning tool useful for when you need to answer logic-based questions."

    llm = get_llm(temperature=0)

    reasoning_prompt_template = """
    You are a reasoning agent tasked with solving the student's logic-based questions.
    Logically arrive at the solution, and be factual. In your answers, clearly detail the steps involved and give
    the final answer. Provide the response in bullet points. Student query: "{student_query}"
    """

    reasoning_prompt = PromptTemplate(
        input_variables=["student_query"],
        template=reasoning_prompt_template
    )

    @chain
    def reasoning_chain(inputs: dict) -> str | list[str] | dict:
        response = llm.invoke([
            HumanMessage(
                content=[
                    {"type": "text", "text": inputs["prompt"]},
                ]
            )
        ])

        return response.content

    return reasoning_chain.invoke({
        "prompt": reasoning_prompt.format_prompt(
           student_query=student_query
        ).to_string()
    })
