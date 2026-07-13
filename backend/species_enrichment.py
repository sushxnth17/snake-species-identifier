import os
import json
import logging
import re
from typing import Optional

from backend.config import settings
from backend.schemas import SpeciesEnrichment
from backend.groq_client import get_groq_client
from backend.metadata import SNAKE_METADATA
from backend.metrics import metrics_tracker

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join("data", "species_enrichment")

def get_cache_key(species_name: str) -> str:
    """
    Generates a safe normalized cache filename from a species label.
    """
    normalized = species_name.lower().strip()
    # Remove any character that is not alphanumeric, underscore, or hyphen
    normalized = re.sub(r'[^a-z0-9_\-]', '', normalized.replace(' ', '_'))
    return f"{normalized}.json"

def get_species_enrichment(species_label: str) -> Optional[SpeciesEnrichment]:
    """
    Retrieves herpetological enrichment information for a validated species.
    Checks local JSON cache first; falls back to Groq API.
    """
    # 1. Prompt Injection resistance & species validation
    normalized_label = species_label.lower().strip()
    if normalized_label not in SNAKE_METADATA or normalized_label in ("uncertain", "uncertain prediction"):
        logger.warning(f"Enrichment requested for unsupported/untrusted species: {species_label}")
        return None
        
    common_name = SNAKE_METADATA[normalized_label]["common_name"]
    cache_filename = get_cache_key(normalized_label)
    cache_path = os.path.join(CACHE_DIR, cache_filename)
    
    # 2. Check local cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            validated_enrichment = SpeciesEnrichment(**cached_data)
            logger.info(f"Enrichment cache hit for species: {normalized_label}")
            
            # Record metrics
            metrics_tracker.record_enrichment(hit=True, success=True)
            return validated_enrichment
        except Exception as e:
            logger.warning(f"Corrupted or invalid cache file found at {cache_path}: {e}. Ignoring and continuing.")
            # Do not crash; we will fall back to querying Groq
            
    # 3. Retrieve Groq client
    client = get_groq_client()
    if client is None:
        logger.info(f"Groq client unavailable; species enrichment omitted for {normalized_label}.")
        # Record metrics (cache miss, unsuccessful enrichment)
        metrics_tracker.record_enrichment(hit=False, success=False)
        return None
        
    logger.info(f"Enrichment cache miss. Querying Groq for species: {common_name}")
    try:
        system_prompt = (
            "You are a herpetologist providing educational herpetological information about the requested snake species.\n"
            "The species name has already been determined. You are NOT identifying an image or determining the species.\n"
            "Return ONLY a valid JSON object matching the requested schema. Do NOT include conversational text or markdown formatting.\n"
            "Do NOT provide medical advice, emergency bite treatment, first-aid steps, or capture/handling instructions.\n"
            "Descriptions must use cautious, general language. Do not claim certainty about the user's photographed snake.\n\n"
            "Schema:\n"
            "{\n"
            "  \"overview\": \"string (2 to 4 sentences, educational herpetology overview)\",\n"
            "  \"habitats\": [\"string (list of 3 to 5 common habitat types, concise)\"],\n"
            "  \"appearance\": [\"string (list of 3 to 5 physical traits. Do not claim these are sufficient for field identification)\"],\n"
            "  \"behavior\": \"string (concise description of typical behavior)\",\n"
            "  \"interesting_facts\": [\"string (list of 2 to 4 interesting herpetological facts)\"]\n"
            "}"
        )
        
        user_message = f"Species: {common_name}"
        
        # Enforce json_object response format
        completion = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        response_content = completion.choices[0].message.content
        if not response_content:
            logger.error(f"Received empty response from Groq for {normalized_label}")
            metrics_tracker.record_enrichment(hit=False, success=False)
            return None
            
        parsed_json = json.loads(response_content)
        validated_enrichment = SpeciesEnrichment(**parsed_json)
        
        # 4. Save to cache
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(validated_enrichment.model_dump(), f, ensure_ascii=False, indent=2)
            logger.info(f"Cached validated enrichment for species: {normalized_label}")
        except Exception as cache_err:
            logger.error(f"Failed to write enrichment cache for {normalized_label}: {cache_err}")
            
        # Record metrics
        metrics_tracker.record_enrichment(hit=False, success=True)
        return validated_enrichment
        
    except Exception as e:
        logger.error(f"Groq species enrichment failed for {normalized_label}: {e}")
        # Record metrics
        metrics_tracker.record_enrichment(hit=False, success=False)
        return None
