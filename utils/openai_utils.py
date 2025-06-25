"""
OpenAI utility functions for MoodDecode API
Handles all interactions with OpenAI's GPT-4o model
"""

import os
import asyncio
from typing import Optional
import logging
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Configuration constants
MODEL_NAME = "gpt-4o"
MAX_TOKENS = 150  # Conservative limit for most responses
TEMPERATURE = 0.3  # Low temperature for consistent, focused responses
REQUEST_TIMEOUT = 30  # Timeout in seconds

class OpenAIError(Exception):
    """Custom exception for OpenAI-related errors"""
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
async def call_openai(
    prompt: str,
    model: str = MODEL_NAME,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    system_message: Optional[str] = None
) -> str:
    """
    Make an async call to OpenAI's ChatCompletion API with retry logic
    
    Args:
        prompt (str): The user prompt/question
        model (str): OpenAI model to use (default: gpt-4o)
        max_tokens (int): Maximum tokens in response
        temperature (float): Sampling temperature (0-1)
        system_message (str, optional): System message to set context
    
    Returns:
        str: The AI's response text
        
    Raises:
        OpenAIError: If API call fails after retries
    """
    
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    # Validate API key
    if not os.getenv("OPENAI_API_KEY"):
        raise OpenAIError("OpenAI API key not configured")
    
    try:
        # Prepare messages
        messages = []
        
        if system_message:
            messages.append({
                "role": "system",
                "content": system_message
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        logger.info(f"Making OpenAI API call with model: {model}")
        logger.debug(f"Prompt: {prompt[:200]}...")  # Log first 200 chars for debugging
        
        # Make the API call
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=REQUEST_TIMEOUT
        )
        
        # Extract and validate response
        if not response.choices:
            raise OpenAIError("No response choices returned from OpenAI")
        
        content = response.choices[0].message.content
        
        if not content:
            raise OpenAIError("Empty response content from OpenAI")
        
        # Log usage statistics
        if hasattr(response, 'usage'):
            logger.info(f"OpenAI API usage - Prompt tokens: {response.usage.prompt_tokens}, "
                       f"Completion tokens: {response.usage.completion_tokens}, "
                       f"Total tokens: {response.usage.total_tokens}")
        
        logger.info("OpenAI API call successful")
        return content.strip()
        
    except asyncio.TimeoutError:
        error_msg = f"OpenAI API call timed out after {REQUEST_TIMEOUT} seconds"
        logger.error(error_msg)
        raise OpenAIError(error_msg)
        
    except Exception as e:
        error_msg = f"OpenAI API call failed: {str(e)}"
        logger.error(error_msg)
        raise OpenAIError(error_msg)

async def call_openai_with_system_context(
    prompt: str,
    system_context: str,
    **kwargs
) -> str:
    """
    Convenience function to call OpenAI with a system message
    
    Args:
        prompt (str): User prompt
        system_context (str): System message for context
        **kwargs: Additional arguments for call_openai
    
    Returns:
        str: AI response
    """
    return await call_openai(
        prompt=prompt,
        system_message=system_context,
        **kwargs
    )

async def validate_openai_connection() -> bool:
    """
    Test the OpenAI API connection
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        test_response = await call_openai(
            prompt="Say 'API connection successful'",
            max_tokens=10
        )
        return "successful" in test_response.lower()
    except Exception as e:
        logger.error(f"OpenAI connection validation failed: {str(e)}")
        return False

# Specialized functions for each endpoint
async def analyze_emotion(text: str) -> str:
    """
    Specialized function for emotion analysis
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: Detected emotion
    """
    system_message = """You are an expert emotion analyzer. Your task is to identify the primary emotion in text.
    Respond with only one word representing the main emotion. Use common emotions like:
    happy, sad, angry, anxious, excited, frustrated, neutral, confused, hopeful, disappointed, fearful, surprised."""
    
    prompt = f"Analyze the emotion in this text: '{text}'"
    
    return await call_openai_with_system_context(
        prompt=prompt,
        system_context=system_message,
        max_tokens=10,
        temperature=0.1
    )

async def detect_crisis_indicators(text: str) -> bool:
    """
    Specialized function for crisis detection
    
    Args:
        text (str): Text to analyze for crisis indicators
        
    Returns:
        bool: True if crisis detected, False otherwise
    """
    system_message = """You are a mental health crisis detection system. Analyze text for signs of:
    - Suicidal ideation or self-harm
    - Extreme hopelessness or despair
    - Direct threats to self
    - Expressions of wanting to die or not live
    
    Respond with only 'true' if ANY crisis indicators are present, 'false' if none are detected.
    Be sensitive but thorough in your analysis."""
    
    prompt = f"Analyze this text for crisis indicators: '{text}'"
    
    response = await call_openai_with_system_context(
        prompt=prompt,
        system_context=system_message,
        max_tokens=5,
        temperature=0.0  # Zero temperature for consistent crisis detection
    )
    
    return response.lower().strip() == "true"

async def generate_summary(text: str) -> str:
    """
    Specialized function for text summarization
    
    Args:
        text (str): Text to summarize
        
    Returns:
        str: Generated summary
    """
    system_message = """You are a professional text summarizer. Create concise, accurate summaries that:
    - Capture the main points and key information
    - Are clear and readable
    - Maintain the original meaning
    - Are significantly shorter than the original text"""
    
    prompt = f"Summarize this text: '{text}'"
    
    return await call_openai_with_system_context(
        prompt=prompt,
        system_context=system_message,
        max_tokens=200,
        temperature=0.2
    )