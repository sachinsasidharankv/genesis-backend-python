from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage
from langchain_core.runnables import chain

from src.utils import get_llm


memory = ConversationBufferMemory(memory_key="history", input_key="student_query")


def run_teacher_chain(
    student_query: str,
    highlighted_text: str,
    reference_page_base64: str
) -> str:
    llm = get_llm(use_groq=True, temperature=0)

    teaching_prompt_template = """
    You are a teaching agent tasked with teaching a student about a particular topic.
    The student has highlighted a particular text in a PDF page.
    You need to answer the student's query strictly based on this highlighted text and the given PDF page.

    The highlighted text: "{highlighted_text}"

    Previous conversation history: "{history}"
    Student query: {student_query}
    """

    teaching_prompt = PromptTemplate(
        input_variables=["student_query", "highlighted_text", "history"],
        template=teaching_prompt_template
    )

    @chain
    def teaching_chain(inputs: dict) -> str | list[str] | dict:
        response = llm.invoke([
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": inputs["prompt"]
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{inputs['reference_page_base64']}"
                        }
                    }
                ]
            )
        ])

        memory.save_context({"student_query": inputs["student_query"]}, {"response": response.content})

        return response.content

    conversation_history = memory.load_memory_variables({"student_query": student_query})["history"]

    return teaching_chain.invoke({
        "prompt": teaching_prompt.format_prompt(
            student_query=student_query,
            highlighted_text=highlighted_text,
            history=conversation_history
        ).to_string(),
        "reference_page_base64": reference_page_base64
    })
