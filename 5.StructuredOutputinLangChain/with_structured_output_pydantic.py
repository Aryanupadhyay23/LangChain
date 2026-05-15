from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from typing import Optional, TypedDict, Annotated, Literal
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Get API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)

# Define structured schema
class Review(BaseModel):
    key_themes: list[str] = Field(description="The main themes discussed in the review")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="The sentiment of the review")
    pros: Optional[list[str]] = Field(description="The positive aspects mentioned in the review")
    cons: Optional[list[str]] = Field(description="The negative aspects mentioned in the review")
    name: Annotated[Optional[str], "The name of the movie being reviewed"]

# Convert model into structured output model
structured_model = model.with_structured_output(Review)

# Invoke model
result = structured_model.invoke(
    """
    Interstellar is a visually stunning and emotionally powerful science fiction film directed by Christopher Nolan. 
    The movie combines space exploration, time dilation, human survival, and emotional storytelling into a deeply immersive experience.

    The cinematography and visual effects are exceptional, especially the depiction of black holes, distant planets, and space travel. 
    Matthew McConaughey delivers one of his strongest performances, bringing emotional depth to the character of Cooper. 
    The soundtrack by Hans Zimmer greatly enhances the intensity and emotional weight of the film.

    The movie can feel complex at times because of its scientific concepts and nonlinear time effects, 
    but it rewards viewers with a meaningful and thought-provoking narrative about love, sacrifice, and humanity’s future.
    """
)

print(result)
