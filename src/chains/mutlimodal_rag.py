import json
from byaldi import RAGMultiModalModel
from pdf2image import convert_from_path
from langchain_core.runnables import chain
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from src.utils import get_llm, pil_image_to_base64
from src.models import SubtopicListModel


def preprocess_pdf(filepath, index_name):
    rag_model = RAGMultiModalModel.from_pretrained("vidore/colpali")

    vision_model = get_llm()

    rag_model.index(
        input_path=filepath,
        index_name=index_name,
        store_collection_with_index=False,
        overwrite=True
    )

    images = get_relevant_pdf_pages(
        search_query="table of contents, index",
        filepath=filepath,
        top_k=5
    )

    images_base64 = []
    for image in images:
        images_base64.append(pil_image_to_base64(image))

    parser = JsonOutputParser(pydantic_object=SubtopicListModel)

    @chain
    def get_subtopics(inputs: dict) -> str | list[str] | dict:
        """Invoke model with image and prompt."""
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": inputs["prompt"]},
                    {"type": "text", "text": parser.get_format_instructions()},
                ]
            ),
        ]

        for image_base64 in inputs["images_base64"]:
            messages[0].content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            )

        response = vision_model.invoke(messages)
        return response.content

    vision_chain = get_subtopics | parser

    get_subtopics_prompt = """
    From the given table of contents pages, identify the subtopics and their corresponding page numbers.
    Be careful to select only relevant subtopics, because we will be using these subtopics to generate question papers later.
    """

    subtopics = vision_chain.invoke({
        "prompt": get_subtopics_prompt,
        "images_base64": images_base64
    })

    return json.dumps(subtopics)


def get_relevant_pdf_pages(search_query, filepath, top_k=5):
    rag_model = RAGMultiModalModel.from_index(index_path="physics.pdf")

    results = rag_model.search(search_query, k=top_k)

    print("RAG Results:")
    for result in results:
        print(f"{result.page_num}")

    images = convert_from_path(filepath)
    final_images = []

    for result in results:
        final_images.append(images[result.page_num - 1])

    return final_images
