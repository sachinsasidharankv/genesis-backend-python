import os

from dotenv import load_dotenv
import langchain.globals


load_dotenv()

is_debug = os.environ.get("DEBUG_MODE") == "true"
langchain.globals.set_debug(is_debug)
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
SUBTOPICS_DIR = os.environ.get("SUBTOPICS_DIR", "subtopics")
SESSION_ID = os.environ.get("SESSION_ID", "test")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SUBTOPICS_DIR, exist_ok=True)


if __name__ == "__main__":
    option = int(input(
        "\n1. Preprocessing\n2. Agent\nEnter your option: "))

    if option == 1:
        from src.chains import preprocess_pdf

        while True:
            student_input = input("\nEnter PDF filename: ")
            subtopics = preprocess_pdf(
                filepath=f"{UPLOAD_DIR}/{student_input}",
                index_name=student_input
            )

            filename = f"{SUBTOPICS_DIR}/{student_input.split('.')[0]}_subtopics.json"
            with open(filename, 'w', encoding='utf-8') as json_file:
                json_file.write(subtopics)

    elif option == 2:
        from src.agent import get_mars_agent

        mars_agent = get_mars_agent(session_id=SESSION_ID)
        while True:
            student_input = input("\nStudent: ")
            response = mars_agent.invoke({
                "input": student_input
            },
                config={"configurable": {"session_id": SESSION_ID}},
            )
            print(f"Agent: {response['output']}")

    else:
        print("Exiting.")
