import json
import os
from langchain.agents import tool
from langchain.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from src.models import QPSubtopicModel, QuestionModel
from src.utils import get_llm, pil_image_to_base64
from src.constants import exam, subtopics
from src.chains.mutlimodal_rag import get_relevant_pdf_pages


UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")


@tool(return_direct=True)
def qp_generation_tool(
    student_query: str,
    student_summary: str,
    num_questions: int,
    time: int
) -> str:
    """Tool used to generate a question paper based on student query.
    Number of questions and time are given by the user.
    The test can also be generated without a timer.
    """
    llm = get_llm(temperature=0)

    identify_subtopics_prompt_template = """
    You are tasked to identify relevant subtopics for a student preparing for {exam}.
    The subtopics should be selected from the following list based on the student's query and their past {exam} performance summary.
    You are allowed to select more than one subtopics from the list.
    The student's query is: "{student_query}".
    The student's performance summary is: "{student_summary}".
    Select the subtopics that are relevant to this query from the list below:

    Subtopics: {subtopics}
    
    Also modify the student query to effectively search in a RAG system to find relevant PDF pages based on the selected subtopics.
    The modified query should strictly avoid common words and should only have words that can help retrieve relevant pages from a PDF.
    """

    identify_subtopics_prompt = PromptTemplate(
        input_variables=[
            "exam",
            "student_query",
            "student_summary",
            "subtopics"
        ],
        template=identify_subtopics_prompt_template
    )

    subtopic_parser = JsonOutputParser(pydantic_object=QPSubtopicModel)

    @chain
    def generate_subtopics(inputs: dict) -> str | list[str] | dict:
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": inputs["prompt"]},
                    {"type": "text", "text": subtopic_parser.get_format_instructions()},
                ]
            )
        ]

        response = llm.invoke(messages)
        return response.content

    subtopic_chain = generate_subtopics | subtopic_parser

    subtopic_result = subtopic_chain.invoke(
        {
            "prompt": identify_subtopics_prompt.format_prompt(
                exam=exam,
                student_query=student_query,
                subtopics=", ".join(subtopics),
                student_summary=student_summary
            ).to_string(),
        }
    )

    identified_subtopics = subtopic_result["subtopics"]
    modified_student_query = subtopic_result["modified_student_query"]

    relevant_pages = get_relevant_pdf_pages(
        search_query=modified_student_query,
        filepath=f"{UPLOAD_DIR}/0.pdf"
    )

    pages_base64 = []
    for page in relevant_pages:
        pages_base64.append(pil_image_to_base64(page))

    qp_generation_prompt_template = """
    You are tasked to generate multiple-choice questions for a student preparing for {exam} for {time} seconds.
    The generated questions should be strictly based on the pages given from the syllabus.
    The subtopics selected for preparing the questions are: "{identified_subtopics}"
    The student has given a query: "{student_query}".

    Based on the selected subtopics, prepare questions based on the student query.
    You have to strictly generate {num_questions} questions.

    For each question, provide:
    - The question (difficulty should be based on {exam}).
    - 4 answer options (follow the same pattern as of {exam}).
    - The correct answer (the list index of the correct option).
    - The difficulty of the question (easy, medium or hard).
    - Total time allowed (time for answering this particular question (in seconds) based on {exam}) (set time = 0 if question paper is without a timer).
    - The related subtopics (taken from the give subtopic list).
    - A detailed explanation of the correct answer (should be very detailed containing everything that the student requires to answer this question, as we require this to clarify the student's doubts).

    A summary of the student's preferences and past performances are also given below.
    Use the summary to make proper questions (by changing the difficulty for the questions) according to the student's current level of performance.
    Summary: "{student_summary}"
    """

    qp_generation_prompt = PromptTemplate(
        input_variables=[
            "num_questions",
            "exam",
            "time",
            "student_query",
            "identified_subtopics",
            "student_summary"
        ],
        template=qp_generation_prompt_template
    )

    qp_parser = JsonOutputParser(pydantic_object=QuestionModel)

    @chain
    def generate_mcqs(inputs: dict) -> str | list[str] | dict:
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": inputs["prompt"]},
                    {"type": "text", "text": qp_parser.get_format_instructions()},
                ]
            )
        ]

        for page_base64 in inputs["pages_base64"]:
            messages[0].content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{page_base64}"
                    }
                }
            )

        response = llm.invoke(messages)
        return response.content

    qp_generation_chain = generate_mcqs | qp_parser

    return json.dumps(qp_generation_chain.invoke({
        "prompt": qp_generation_prompt.format_prompt(
            student_query=student_query,
            num_questions=num_questions,
            time=time,
            exam=exam,
            student_summary=student_summary,
            identified_subtopics=", ".join(identified_subtopics)
        ).to_string(),
        "pages_base64": pages_base64
    }))
