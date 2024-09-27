from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor

from src.utils import get_llm
from src.tools import (
    action_tool,
    qp_generation_tool,
    feedback_tool,
    evaluation_tool,
    teaching_tool,
    reasoning_tool,
    wolfram_alpha_tool,
    wikipedia_tool,
    whatsapp_tool
)


def get_our_agent(memory=None):
    llm = get_llm(temperature=0)

    tools = [
        action_tool,
        qp_generation_tool,
        feedback_tool,
        evaluation_tool,
        teaching_tool,
        wikipedia_tool,
        wolfram_alpha_tool,
        reasoning_tool,
        whatsapp_tool
    ]

    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_openai_tools_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        handle_parsing_errors=True
    )

    return agent_executor
