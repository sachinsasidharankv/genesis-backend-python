from langchain_community.utilities.twilio import TwilioAPIWrapper
from langchain.agents import tool


@tool
def whatsapp_tool(
    message_body: str,
    to_number: str
) -> str:
    """A useful tool for sending messages in WhatsApp."""

    twilio = TwilioAPIWrapper()
    return twilio.run(
        body=message_body,
        to=to_number
    )
