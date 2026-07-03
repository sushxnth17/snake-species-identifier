from typing import Dict, Any

# Static database for snake species metadata.
# Keys are lowercase to match normalized predicted labels.
SNAKE_METADATA: Dict[str, Dict[str, Any]] = {
    "cobra": {
        "scientific_name": "Naja naja",
        "venomous": True,
        "first_aid": (
            "1. Keep the victim calm and reassured to slow venom circulation.\n"
            "2. Immobilize the bitten limb using a splint or bandage (do not wrap too tight).\n"
            "3. Remove rings, bracelets, or tight clothing near the bite area.\n"
            "4. Transport the victim immediately to the nearest medical facility with anti-venom.\n"
            "5. DO NOT cut the bite site, apply a tourniquet, or try to suck out the venom."
        )
    },
    "krait": {
        "scientific_name": "Bungarus caeruleus",
        "venomous": True,
        "first_aid": (
            "1. Keep the victim completely still; krait bites can be painless but are highly neurotoxic.\n"
            "2. Apply a broad pressure immobilization bandage over the entire bitten limb.\n"
            "3. Seek emergency medical attention immediately. Keep breathing airways clear.\n"
            "4. DO NOT wash the bite site (venom residue can help identify the snake later).\n"
            "5. DO NOT cut the wound, use ice, or apply tight tourniquets."
        )
    }
}

def get_snake_metadata(species_name: str) -> Dict[str, Any]:
    """
    Retrieves safety metadata for a given snake species by name.
    
    Args:
        species_name: The predicted snake species name.
        
    Returns:
        A dictionary containing scientific name, venom status, and first-aid instructions.
    """
    normalized_name = species_name.lower().strip()
    return SNAKE_METADATA.get(normalized_name, {
        "scientific_name": "Unknown",
        "venomous": False,
        "first_aid": "Seek general medical attention if bitten by any unidentified snake species."
    })
