from langchain.prompts import PromptTemplate
from langchain.agents import tool
from langchain_core.messages import SystemMessage
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser

from src.models import FeedbackModel
from src.utils import get_llm


@tool
def feedback_tool(exam_results: dict, student_summary: str) -> str:
    """Tool used to generate feedback after analysing the results of an exam the student has taken"""
    llm = get_llm(temperature=0.5)

    feedback_prompt_template = """
    You are a feedback generation tool tasked with generating feedback for a student's exam performance.
    You should give a one-line feedback on each question, a one-line feedback on the whole exam and a one-line suggestion on how to improve.

    The results of the exam are given below: 
    {exam_results}

    A summary of the student's preferences and past performances are also given below.
    Use the summary to give suggestions on how to improve in upcoming exams.
    Summary: "{student_summary}"
    """

    feedback_prompt = PromptTemplate(
        input_variables=["exam_results", "student_summary"],
        template=feedback_prompt_template
    )

    @chain
    def generate_feedback(inputs: dict) -> str | list[str] | dict:
        response = llm.invoke([
            SystemMessage(
                content=[
                    {"type": "text", "text": inputs["prompt"]},
                ]
            )
        ])

        return response.content

    parser = JsonOutputParser(pydantic_object=FeedbackModel)

    feedback_chain = generate_feedback | parser

    return feedback_chain.invoke({
        "prompt": feedback_prompt.format_prompt(
           exam_results=exam_results,
           student_summary=student_summary
        ).to_string()
    })
