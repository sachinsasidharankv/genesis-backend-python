from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.agents import tool


@tool
def wolfram_alpha_tool(
    student_query: str
) -> str:
    """Tool useful for when you need to answer numeric questions.
    This tool is only for numeric questions and nothing else."""

    wolfram_alpha = WolframAlphaAPIWrapper()
    return wolfram_alpha.run(
        query=student_query
    )
