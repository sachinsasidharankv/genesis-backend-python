from langchain.prompts import PromptTemplate
from langchain.agents import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import chain

from src.utils import get_llm


@tool
def evaluation_tool(
    student_query: str,
    student_summary: str,
    question_dict: str,
) -> str:
    "Teaching tool useful for teaching a student strictly based on a specific question."

    llm = get_llm(temperature=0)

    evaluating_prompt_template = """
    You are a evaluation plus teaching agent tasked with evaluating a student's answer to a specific question and clarifying the student's queries.
    The question, its options, correct answer, student's answer, subtopics from which the question was taken, detailed explanation, etc. are given below.
    You need to answer the student's query strictly based on this specific question and its information.
    Be careful not to give the answer directly but guide the student to learn the topics by breaking down the question to simpler subproblems and asking the student to solve them.
    You are allowed to give the detailed explanation only when you are convinced that the user is struggling to answer the question.

    The question and its related information: "{question_dict}"
    A summary of the student is also given: "{student_summary}"

    Strictly remember not to answer anything other than any doubts related to the given question.

    Student query: {student_query}
    """

    evaluating_prompt = PromptTemplate(
        input_variables=["student_query", "question_dict"],
        template=evaluating_prompt_template
    )

    @chain
    def evaluating_chain(inputs: dict) -> str | list[str] | dict:
        response = llm.invoke([
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": inputs["prompt"]
                    },
                ]
            )
        ])

        return response.content

    return evaluating_chain.invoke({
        "prompt": evaluating_prompt.format_prompt(
            student_query=student_query,
            student_summary=student_summary,
            question_dict=question_dict,
        ).to_string()
    })
