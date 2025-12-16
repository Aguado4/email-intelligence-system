"""
Email classification workflow using LangGraph.

This module implements a state machine that:
1. Classifies emails into categories
2. Routes based on confidence scores
3. Re-analyzes low-confidence classifications

Supports multiple LLM providers: Gemini, OpenAI, Anthropic
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
import json
import logging

from config.settings import get_settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


# ============= State Definition =============

class EmailClassificationState(TypedDict):
    """
    State that flows through the workflow.
    
    TypedDict provides type hints for the state dictionary,
    enabling better IDE support and catching errors early.
    """
    # Input fields
    email_id: str
    subject: str
    body: str
    sender: str
    
    # Processing fields
    category: str
    confidence: float
    reasoning: str
    keywords: list[str]
    
    # Metadata
    retry_count: int
    processing_stage: str


# ============= LLM Factory =============

def get_llm() -> BaseChatModel:
    """
    Factory function to create LLM instance based on provider.
    
    This abstraction allows switching providers without changing
    the workflow logic - just update .env file.
    
    Returns:
        Configured LLM instance for the selected provider
        
    Raises:
        ValueError: If provider is not supported
    """
    provider = settings.llm_provider
    api_key = settings.get_api_key()
    
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.model_name,
            google_api_key=api_key,
            max_output_tokens=settings.max_tokens,
            temperature=settings.temperature
        )
    
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.model_name,
            openai_api_key=api_key,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature
        )
    
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.model_name,
            anthropic_api_key=api_key,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature
        )
    
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: gemini, openai, anthropic"
        )


# ============= Helper Functions =============

def clean_json_response(content: str) -> str:
    """
    Clean LLM response to extract JSON.
    
    LLMs sometimes wrap JSON in markdown code blocks.
    This function removes those wrappers.
    
    Args:
        content: Raw LLM response
        
    Returns:
        Cleaned JSON string
    """
    content = content.strip()
    
    # Remove markdown code blocks
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    
    if content.endswith("```"):
        content = content[:-3]
    
    return content.strip()


# ============= Node Functions =============

async def classify_email_node(state: EmailClassificationState) -> EmailClassificationState:
    """
    Primary classification node.
    
    This node:
    1. Constructs a classification prompt
    2. Calls the LLM
    3. Parses structured output
    4. Updates state with results
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with classification results
    """
    logger.info(f"Classifying email {state['email_id']}")
    
    # Construct prompt with clear instructions
    system_prompt = """You are an expert email classifier. Analyze emails and classify them into exactly one category:

- spam: Unsolicited emails, phishing attempts, scams, promotional bulk mail
- important: Urgent business matters, critical notifications, time-sensitive requests
- neutral: Regular correspondence, newsletters, non-urgent communication

You MUST respond with valid JSON only, no additional text or explanation:
{
    "category": "spam" | "important" | "neutral",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation (1-2 sentences)",
    "keywords": ["key", "terms", "that", "influenced", "decision"]
}"""

    user_prompt = f"""Classify this email:

Subject: {state['subject']}
From: {state['sender']}
Body: {state['body'][:1000]}"""  # Limit body length

    # Call LLM
    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        
        # Parse JSON response
        content = clean_json_response(response.content)
        result = json.loads(content)
        
        # Validate required fields
        required_fields = ["category", "confidence", "reasoning", "keywords"]
        if not all(field in result for field in required_fields):
            raise ValueError(f"Missing required fields in response: {result}")
        
        # Validate category
        valid_categories = ["spam", "important", "neutral"]
        if result["category"] not in valid_categories:
            raise ValueError(f"Invalid category: {result['category']}")
        
        # Update state
        return {
            **state,
            "category": result["category"],
            "confidence": float(result["confidence"]),
            "reasoning": result["reasoning"],
            "keywords": result["keywords"],
            "processing_stage": "classified"
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}. Response was: {response.content[:200]}")
        return {
            **state,
            "category": "neutral",
            "confidence": 0.3,
            "reasoning": "Unable to parse LLM response",
            "keywords": [],
            "processing_stage": "error_json_parse"
        }
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return {
            **state,
            "category": "neutral",
            "confidence": 0.3,
            "reasoning": f"Error during classification: {str(e)[:100]}",
            "keywords": [],
            "processing_stage": "error"
        }


async def reanalyze_node(state: EmailClassificationState) -> EmailClassificationState:
    """
    Re-analysis node for low-confidence classifications.
    
    This node is triggered when initial confidence is low.
    It uses a different prompting strategy with more context.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with re-analyzed results
    """
    logger.info(f"Re-analyzing email {state['email_id']} (retry {state['retry_count']})")
    
    # Enhanced prompt with previous classification context
    system_prompt = """You are an expert email classifier performing a SECOND analysis.

The first classification was uncertain. Please carefully reconsider:

Categories:
- spam: Unsolicited, promotional, or malicious emails
- important: Time-sensitive business matters requiring action
- neutral: Regular correspondence

Provide a more confident assessment. Consider:
1. Sender reputation indicators
2. Urgency keywords
3. Call-to-action presence
4. Professional vs promotional language

You MUST respond with valid JSON only:
{
    "category": "spam" | "important" | "neutral",
    "confidence": 0.0 to 1.0,
    "reasoning": "Detailed explanation referencing specific indicators",
    "keywords": ["specific", "indicators", "found"]
}"""

    user_prompt = f"""Re-analyze this email with more scrutiny:

Subject: {state['subject']}
From: {state['sender']}
Body: {state['body']}

Previous classification: {state['category']} (confidence: {state['confidence']:.2f})
Previous reasoning: {state['reasoning']}"""

    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        content = clean_json_response(response.content)
        result = json.loads(content)
        
        # Validate
        valid_categories = ["spam", "important", "neutral"]
        if result["category"] not in valid_categories:
            raise ValueError(f"Invalid category: {result['category']}")
        
        return {
            **state,
            "category": result["category"],
            "confidence": float(result["confidence"]),
            "reasoning": result["reasoning"],
            "keywords": result["keywords"],
            "retry_count": state["retry_count"] + 1,
            "processing_stage": "reanalyzed"
        }
        
    except Exception as e:
        logger.error(f"Re-analysis error: {e}")
        # Keep original classification but mark as uncertain
        return {
            **state,
            "retry_count": state["retry_count"] + 1,
            "processing_stage": "reanalysis_failed"
        }


# ============= Conditional Edge Logic =============

def should_reanalyze(state: EmailClassificationState) -> str:
    """
    Decision function for conditional routing.
    
    This determines which node to execute next based on:
    - Confidence score
    - Number of retries already attempted
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name: "reanalyze", "end", or "max_retries"
    """
    confidence = state["confidence"]
    retry_count = state.get("retry_count", 0)
    
    # Maximum 1 retry to avoid infinite loops
    if retry_count >= 1:
        logger.info(f"Max retries reached for {state['email_id']}")
        return "max_retries"
    
    # Low confidence triggers re-analysis
    if confidence < settings.low_confidence_threshold:
        logger.info(f"Low confidence ({confidence:.2f}), triggering re-analysis")
        return "reanalyze"
    
    # High confidence, we're done
    logger.info(f"Sufficient confidence ({confidence:.2f}), classification complete")
    return "end"


# ============= Build Workflow =============

def create_classification_workflow() -> StateGraph:
    """
    Construct the LangGraph workflow.
    
    Workflow structure:
        START → classify → [decision] → reanalyze → END
                              ↓
                             END
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize graph with state schema
    workflow = StateGraph(EmailClassificationState)
    
    # Add nodes
    workflow.add_node("classify", classify_email_node)
    workflow.add_node("reanalyze", reanalyze_node)
    
    # Set entry point
    workflow.set_entry_point("classify")
    
    # Add conditional edges from classify node
    workflow.add_conditional_edges(
        "classify",
        should_reanalyze,
        {
            "reanalyze": "reanalyze",
            "end": END,
            "max_retries": END
        }
    )
    
    # Reanalyze always goes to END (no more retries)
    workflow.add_edge("reanalyze", END)
    
    return workflow.compile()


# ============= Main Execution Function =============

async def classify_email(
    email_id: str,
    subject: str,
    body: str,
    sender: str
) -> dict:
    """
    Main entry point for email classification.
    
    This function:
    1. Initializes state
    2. Executes workflow
    3. Returns final classification
    
    Args:
        email_id: Unique email identifier
        subject: Email subject line
        body: Email body content
        sender: Sender email address
        
    Returns:
        Dictionary with classification results
    """
    logger.info(f"Starting classification for email {email_id} using {settings.llm_provider}")
    
    # Initialize state
    initial_state = EmailClassificationState(
        email_id=email_id,
        subject=subject,
        body=body,
        sender=sender,
        category="",
        confidence=0.0,
        reasoning="",
        keywords=[],
        retry_count=0,
        processing_stage="initialized"
    )
    
    # Create and run workflow
    workflow = create_classification_workflow()
    final_state = await workflow.ainvoke(initial_state)
    
    # Return only relevant fields
    return {
        "email_id": final_state["email_id"],
        "category": final_state["category"],
        "confidence": final_state["confidence"],
        "reasoning": final_state["reasoning"],
        "keywords": final_state["keywords"],
        "processing_stage": final_state["processing_stage"]
    }