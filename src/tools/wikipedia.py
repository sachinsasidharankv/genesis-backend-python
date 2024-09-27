from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain.agents import tool


@tool
def wikipedia_tool(
    student_query: str
) -> str:
    """A useful tool for searching the Internet.
    Used to find information on world events, issues, dates, years, etc.
    Worth using for general topics. Use precise questions."""

    wikipedia = WikipediaAPIWrapper()
    return wikipedia.run(
        query=student_query
    )
