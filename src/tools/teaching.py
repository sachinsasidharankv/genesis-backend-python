from langchain.prompts import PromptTemplate
from langchain.agents import tool
from langchain_core.messages import SystemMessage
from langchain_core.runnables import chain

from src.utils import get_llm


@tool
def teaching_tool(
    student_query: str,
    highlighted_text: str,
    reference_page_base64: str
) -> str:
    "Teaching tool useful for teaching a student strictly based on syllabus."

    llm = get_llm(temperature=0)

    reasoning_prompt_template = """
    You are a teaching agent tasked with teaching a student about a particular topic.
    The student has highlighted a particular text in a PDF page.
    You need to answer the student's query strictly based on this highlighted text and the given PDF page.

    The highlighted text: "{highlighted_text}"

    Strictly remember not to answer anything other than any doubts related to the given PDF page.

    Student query: {student_query}
    """

    reasoning_prompt = PromptTemplate(
        input_variables=["student_query", "highlighted_text"],
        template=reasoning_prompt_template
    )

    @chain
    def teaching_chain(inputs: dict) -> str | list[str] | dict:
        response = llm.invoke([
            SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": inputs["prompt"]
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{inputs["reference_page_base64"]}"
                        }
                    }
                ]
            )
        ])

        return response.content

    return teaching_chain.invoke({
        "prompt": reasoning_prompt.format_prompt(
            student_query=student_query,
            highlighted_text=highlighted_text,
        ).to_string(),
        "reference_page_base64": reference_page_base64
    })
