import os

from dotenv import load_dotenv
import langchain.globals


load_dotenv()

is_debug = os.environ.get("DEBUG_MODE") == "true"
langchain.globals.set_debug(is_debug)
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
SUBTOPICS_DIR = os.environ.get("SUBTOPICS_DIR", "subtopics")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SUBTOPICS_DIR, exist_ok=True)


if __name__ == "__main__":
    option = int(input(
        "\n1. Preprocessing\n2. QP Generation\n3. Feedback\n4. Teaching\nEnter your option: "))

    if option == 1:
        from src.chains import preprocess_pdf

        while True:
            user_input = input("\nEnter PDF filename: ")
            subtopics = preprocess_pdf(
                filepath=f"{UPLOAD_DIR}/{user_input}",
                index_name=user_input
            )

            filename = f"{SUBTOPICS_DIR}/{user_input.split('.')[0]}_subtopics.json"
            with open(filename, 'w', encoding='utf-8') as json_file:
                json_file.write(subtopics)

    elif option == 2:
        from src.chains import qp_generation

        while True:
            user_input = input("\nEnter query: ")
            print(qp_generation(user_input))

    elif option in (3, 4):
        from src.agent import get_our_agent
        from langchain.memory import ConversationBufferMemory

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        agent_executor = get_our_agent(memory=memory)
        while True:
            context = input("Enter context: ")
            user_input = input("\nEnter your doubt: ")
            chat_history = memory.buffer_as_messages
            response = agent_executor.invoke({
                "input": f"Context: {context}\nStudent query: {user_input}",
                "chat_history": chat_history,
            })
            print(response["output"])

    else:
        print("Exiting.")
