"""
MoodDecode - NLP API for mood analysis, crisis detection, and text summarization
Built with FastAPI and OpenAI GPT-4o
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import logging
from utils.openai_utils import call_openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MoodDecode API",
    description="AI-powered mood analysis, crisis detection, and text summarization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Pydantic models for request/response validation
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Input text to analyze")

class MoodResponse(BaseModel):
    emotion: str = Field(..., description="Detected primary emotion")

class CrisisResponse(BaseModel):
    crisis_detected: bool = Field(..., description="Whether crisis/suicidal ideation is detected")

class SummaryResponse(BaseModel):
    summary: str = Field(..., description="Summarized version of input text")

# Health check endpoint
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "MoodDecode API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "MoodDecode API",
        "version": "1.0.0",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
    }

@app.post("/analyze-mood", response_model=MoodResponse)
async def analyze_mood(input_data: TextInput) -> Dict[str, Any]:
    """
    Analyze the emotional tone of input text
    
    Returns the primary emotion detected in the text such as:
    happy, sad, angry, anxious, excited, frustrated, etc.
    """
    try:
        # Construct prompt for mood analysis
        prompt = f"""
        Analyze the emotional tone of the following text and identify the primary emotion.
        
        Text: "{input_data.text}"
        
        Respond with only ONE word that best describes the primary emotion (e.g., happy, sad, angry, anxious, excited, frustrated, neutral, confused, hopeful, disappointed).
        
        Emotion:"""
        
        logger.info(f"Analyzing mood for text: {input_data.text[:100]}...")
        
        # Call OpenAI API
        emotion = await call_openai(prompt)
        
        # Clean and validate response
        emotion = emotion.strip().lower()
        
        logger.info(f"Detected emotion: {emotion}")
        
        return {"emotion": emotion}
        
    except Exception as e:
        logger.error(f"Error in mood analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze mood: {str(e)}")

@app.post("/detect-crisis", response_model=CrisisResponse)
async def detect_crisis(input_data: TextInput) -> Dict[str, Any]:
    """
    Detect potential crisis situations or suicidal ideation in text
    
    This endpoint identifies if the text contains indicators of:
    - Suicidal thoughts or intentions
    - Self-harm indicators
    - Severe emotional crisis
    """
    try:
        # Construct prompt for crisis detection
        prompt = f"""
        Analyze the following text for signs of suicidal ideation, self-harm, or severe emotional crisis.
        
        Text: "{input_data.text}"
        
        Look for indicators such as:
        - Direct or indirect mentions of suicide or self-harm
        - Expressions of hopelessness or worthlessness
        - Statements about wanting to die or not wanting to live
        - References to suicide methods or plans
        - Extreme despair or feeling trapped
        
        Respond with only "true" if crisis indicators are detected, or "false" if not.
        
        Crisis detected:"""
        
        logger.info(f"Checking for crisis indicators in text: {input_data.text[:100]}...")
        
        # Call OpenAI API
        response = await call_openai(prompt)
        
        # Parse boolean response
        crisis_detected = response.strip().lower() == "true"
        
        logger.info(f"Crisis detected: {crisis_detected}")
        
        # Log crisis detection for monitoring (without sensitive content)
        if crisis_detected:
            logger.warning("Crisis indicators detected in submitted text")
        
        return {"crisis_detected": crisis_detected}
        
    except Exception as e:
        logger.error(f"Error in crisis detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to detect crisis: {str(e)}")

@app.post("/summarize", response_model=SummaryResponse)
async def summarize(input_data: TextInput) -> Dict[str, Any]:
    """
    Generate a concise summary of the input text
    
    Creates a brief, coherent summary that captures the main points
    and key information from the original text.
    """
    try:
        # Construct prompt for summarization
        prompt = f"""
        Summarize the following text in a clear, concise manner. 
        Capture the main points and key information while keeping it brief and readable.
        
        Text: "{input_data.text}"
        
        Summary:"""
        
        logger.info(f"Summarizing text: {input_data.text[:100]}...")
        
        # Call OpenAI API
        summary = await call_openai(prompt)
        
        # Clean response
        summary = summary.strip()
        
        logger.info(f"Generated summary: {summary[:100]}...")
        
        return {"summary": summary}
        
    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to summarize text: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "message": "Please check the API documentation at /docs"}

@app.exception_handler(422)
async def validation_error_handler(request, exc):
    return {"error": "Validation error", "details": exc.detail}

if __name__ == "__main__":
    import uvicorn
    
    # Check if OpenAI API key is configured
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables!")
        exit(1)
    
    # Run the application
    logger.info("Starting MoodDecode API...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )