from pydantic import BaseModel, Field

class ConvertStructure(BaseModel):
    """
    Represents the structure of the output for the conversion/revisor agent.
    """
    generation: str = Field(description="Code that has been generated or revised")

class FeedbackStructure(BaseModel):
    """
    Represents the structure of the output for the reflection agent.
    """
    feedback: str = Field(description="Feedback of the converted/revised code")
    judgement: str = Field(description="Final judgement based on the feedback")
    