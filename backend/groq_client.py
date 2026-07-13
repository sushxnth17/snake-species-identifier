import logging
from typing import Optional, Any

from backend.config import settings

logger = logging.getLogger(__name__)

# Global cached client instance
_groq_client: Optional[Any] = None

def get_groq_client() -> Optional[Any]:
    """
    Retrieves the cached Groq client instance, or lazily initializes it.
    Returns None if Groq enrichment is disabled or the API key is not configured.
    """
    global _groq_client
    if _groq_client is None:
        if not settings.groq_enrichment_enabled:
            logger.info("Groq species enrichment is disabled in configuration.")
            return None
            
        if not settings.groq_api_key:
            logger.warning("GROQ_API_KEY is missing. Groq species enrichment will be unavailable.")
            return None
            
        try:
            import importlib
            groq_module = importlib.import_module("groq")
            Groq = groq_module.Groq
            _groq_client = Groq(api_key=settings.groq_api_key, timeout=settings.groq_timeout)
            logger.info("Groq client initialized successfully.")
        except ImportError:
            logger.error("Failed to import groq SDK. Ensure the 'groq' package is installed.")
            _groq_client = None
        except Exception as e:
            logger.error(f"Error initializing Groq client: {e}")
            _groq_client = None
            
    return _groq_client
