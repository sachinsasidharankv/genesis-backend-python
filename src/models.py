from enum import Enum
from pydantic import BaseModel, Field


class SubtopicModel(BaseModel):
    """Subtopic type"""
    subtopic: str = Field("subtopic identified")
    start_page_number: int = Field("starting page number of the subtopic")
    end_page_number: int = Field("ending page number of the subtopic")


class SubtopicListModel(BaseModel):
    """Subtopics of a PDF"""
    subtopics: list[SubtopicModel] = Field(description="subtopics identified from the PDF pages")


class QPSubtopicModel(BaseModel):
    """Subtopics of a PDF"""
    subtopics: list[str] = Field(description="subtopics selected to generate question paper")
    modified_student_query: str = Field(description="modified student query to search in RAG system of syllabus PDFs")


class Difficulty(str, Enum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class QuestionModel(BaseModel):
    """Question paper generation"""
    question: str = Field(description="generated question")
    options: list[str] = Field(description="multiple choice options for the generated question")
    correct_answer: int = Field(description="the correct option number from the options list")
    difficulty: Difficulty = Field(description="difficulty level of the generated question")
    time: int = Field(description="total time allowed for answering the generated question in seconds")
    subtopics: list[str] = Field(description="related subtopics for the generated question")
    explanation: str = Field("detailed explanation of the question and correct answer")


class FeedbackModel(BaseModel):
    """Feedback generation"""
    overall_feedback: str = Field(description="one-line feedback on the overall performance")
    overall_suggestion: str = Field(description="one-line suggestion on how to improve")
    question_specific_feedback: list[str] = Field(description="one-line feedback on each question")
