"""
Anthropic Claude Client implementation for the memory agent.
"""

import os 
from typing import Dict, List, Optional, Any

from anthropic import Anthropic
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.model.config import get_model_config


# load environment variables
load_dotenv()

def get_claude_client() -> BaseChatModel:
    """
    Initialize and return the Claude language model client.

    Returns:
        BaseChatModel: The configured Claude language model.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    model_name = os.getenv("MODEL_NAME", "claude-3-sonnet-20240229") 
    temperature = float(os.getenv("TEMPERATURE", "0.7"))
    max_tokens = int(os.getenv("MAX_TOKENS", "4096"))  # Increased for memory agent  MODEL_NAME: 


    # initialize the Anthropic client with LangChain
    claude = ChatAnthropic(
        model_name=model_name,
        anthropic_api_key=api_key,
        temperature=temperature,
        max_tokens_to_sample=max_tokens,
    )

    return claude

def get_direct_client() -> Anthropic:
    """
    Get the direct Anthropic client for more advanced usage.

    Returns:
        Anthropic: The Anthropic client.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    return Anthropic(api_key=api_key)

def create_memory_agent_prompt() -> str:
    """
    Create the system prompt for the memory agent.
    
    Returns:
        str: The system prompt for the model.
    """
    return """You are a helpful assistant with memory capabilities.

Your key features:
1. You can remember information shared earlier in the conversation
2. You should use names and personal details that users share with you
3. You should reference past interactions within the current thread
4. Be conversational and helpful

When a user shares personal information:
- Remember details like names, preferences, and interests
- Use these details naturally in future responses
- Make connections between new information and previously shared details

Keep your responses friendly, personable, and relevant to the user's current query while making use of your memory when appropriate.
"""