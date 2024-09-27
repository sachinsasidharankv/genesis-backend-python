from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

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


def get_mars_agent(session_id):
    llm = get_llm(model_name="gpt-4o", use_groq=False, temperature=0)

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
        handle_parsing_errors=True
    )

    memory = ChatMessageHistory(session_id=session_id)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: memory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return agent_with_chat_history
