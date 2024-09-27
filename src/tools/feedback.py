import json
from langchain.prompts import PromptTemplate
from langchain.agents import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser

from src.models import FeedbackModel
from src.utils import get_llm


@tool(return_direct=True)
def feedback_tool(
    exam_results_dict: str,
    student_summary: str
) -> str:
    """Tool used to generate feedback after analysing the results of an exam the student has taken"""

    llm = get_llm(temperature=0)

    feedback_prompt_template = """
    You are a feedback generation tool tasked with generating detailed feedback for a student's exam performance.
    You should give a detailed one-line feedback on the whole exam focusing on different subtopics and a one-line suggestion on how to improve in the weak areas.
    Alsp, give a detailed one-line feedback on each question by analysing how the student answered the question.

    The results of the exam are given below: 
    {exam_results}

    A summary of the student's preferences and past performances are also given below.
    Use the summary to give suggestions on how to improve in upcoming exams.
    Modify the existing summary by adding the strong and weak areas of the student in addition to the student's preferences.
    Student summary: "{student_summary}"
    """

    feedback_prompt = PromptTemplate(
        input_variables=["exam_results", "student_summary"],
        template=feedback_prompt_template
    )

    parser = JsonOutputParser(pydantic_object=FeedbackModel)

    @chain
    def generate_feedback(inputs: dict) -> str | list[str] | dict:
        response = llm.invoke([
            HumanMessage(
                content=[
                    {"type": "text", "text": inputs["prompt"]},
                    {"type": "text", "text": parser.get_format_instructions()},
                ]
            )
        ])

        return response.content

    feedback_chain = generate_feedback | parser

    return json.dumps(feedback_chain.invoke({
        "prompt": feedback_prompt.format_prompt(
            exam_results=exam_results_dict,
            student_summary=student_summary
        ).to_string()
    }))
