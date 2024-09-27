from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage
from langchain_core.runnables import chain

from src.utils import get_llm


memory = ConversationBufferMemory(memory_key="history", input_key="student_query")


def run_guidance_chain(
    student_query: str,
    question_dict: str,
    student_summary: str,
) -> str:
    llm = get_llm(use_groq=True, temperature=0)

    guidance_prompt_template = """
    You are a guidance-focussed teaching agent tasked with evaluating a student's answer to a specific question and clarifying the student's query.
    Be careful not to give the answer directly but guide the student to learn the topics by solving a sub-question.
    Your output should your response to the student in less than 50 words.

    The question and its related information: "{question_dict}"
    A summary of the student is also given: "{student_summary}"
    Previous conversation history: "{history}"
    Student query: "{student_query}"
    """

    guidance_prompt = PromptTemplate(
        input_variables=["student_query", "question_dict"],
        template=guidance_prompt_template
    )

    @chain
    def guidance_chain(inputs: dict) -> str | list[str] | dict:
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

        memory.save_context({"student_query": inputs["student_query"]}, {"response": response.content})

        return response.content

    conversation_history = memory.load_memory_variables({"student_query": student_query})["history"]

    return guidance_chain.invoke({
        "prompt": guidance_prompt.format_prompt(
            student_query=student_query,
            student_summary=student_summary,
            question_dict=question_dict,
            history=conversation_history
        ).to_string(),
        "student_query": student_query
    })
