from typing import Dict, Any

# Static database for snake species metadata.
# Keys are lowercase to match normalized predicted labels.
SNAKE_METADATA: Dict[str, Dict[str, Any]] = {
    "cobra": {
        "common_name": "Spectacled Cobra",
        "scientific_name": "Naja naja",
        "venomous": True,
        "description": "The Indian cobra or spectacled cobra is a highly venomous species of the genus Naja found in the Indian subcontinent. It is characterized by its signature hood and spectacled mark on the back of its neck, and is a member of the 'Big Four' snakes responsible for most bites.",
        "habitat": "Found in forests, wetlands, grasslands, agricultural areas, and near human settlements.",
        "first_aid": (
            "1. Keep the victim calm and reassured to slow venom circulation.\n"
            "2. Immobilize the bitten limb using a splint or loose bandage.\n"
            "3. Remove rings, bracelets, or tight clothing near the bite area.\n"
            "4. Transport the victim immediately to the nearest medical facility with anti-venom.\n"
            "5. DO NOT cut the bite site, apply a tourniquet, or try to suck out the venom."
        ),
        "first_aid_steps": [
            "Keep the victim calm and reassured to slow venom circulation.",
            "Immobilize the bitten limb using a splint or loose bandage.",
            "Remove rings, bracelets, or tight clothing near the bite area.",
            "Transport the victim immediately to the nearest medical facility with anti-venom."
        ],
        "avoid_actions": [
            "DO NOT cut the bite site.",
            "DO NOT apply a tourniquet.",
            "DO NOT try to suck out the venom."
        ]
    },
    "krait": {
        "common_name": "Common Krait",
        "scientific_name": "Bungarus caeruleus",
        "venomous": True,
        "description": "The common krait is a highly venomous species of snake native to the Indian subcontinent. It is nocturnal, secretive, and active mainly at night. Krait bites can be painless and hard to detect, but they carry a highly neurotoxic venom that causes progressive muscular paralysis and respiratory failure.",
        "habitat": "Fields, low scrub forests, agricultural lands, and often enters human dwellings at night in search of food.",
        "first_aid": (
            "1. Keep the victim completely still. Krait bites can be painless but are highly neurotoxic.\n"
            "2. Apply a broad pressure immobilization bandage over the entire bitten limb.\n"
            "3. Seek emergency medical attention immediately. Keep breathing airways clear.\n"
            "4. DO NOT wash the bite site (venom residue can help identify the snake later).\n"
            "5. DO NOT cut the wound, use ice, or apply tight tourniquets."
        ),
        "first_aid_steps": [
            "Keep the victim completely still. Krait bites can be painless but are highly neurotoxic.",
            "Apply a broad pressure immobilization bandage over the entire bitten limb.",
            "Seek emergency medical attention immediately. Keep breathing airways clear."
        ],
        "avoid_actions": [
            "DO NOT wash the bite site (venom residue can help identify the snake later).",
            "DO NOT cut the wound.",
            "DO NOT use ice.",
            "DO NOT apply tight tourniquets."
        ]
    }
}

DEFAULT_METADATA: Dict[str, Any] = {
    "common_name": "Unknown Snake Species",
    "scientific_name": "Unknown",
    "venomous": False,
    "description": "No detailed safety information is available for this species.",
    "habitat": "Unknown",
    "first_aid": "Keep the victim calm, immobilize the bitten limb, and seek emergency medical care immediately.",
    "first_aid_steps": [
        "Keep the victim calm and reassured.",
        "Immobilize the bitten limb.",
        "Seek emergency medical care immediately."
    ],
    "avoid_actions": [
        "DO NOT cut the bite site.",
        "DO NOT apply a tourniquet.",
        "DO NOT try to suck out the venom."
    ]
}

UNCERTAIN_METADATA: Dict[str, Any] = {
    "common_name": "Uncertain Species Identification",
    "scientific_name": "Uncertain",
    "venomous": True,  # Safety-first: treat as potentially venomous
    "description": "The classification confidence is below the reliability threshold. To ensure user safety, this prediction is marked as uncertain. Do not approach or handle the snake.",
    "habitat": "Unknown",
    "first_aid": (
        "1. Treat the snake as venomous out of an abundance of caution.\n"
        "2. Keep the victim calm and reassured to slow venom circulation.\n"
        "3. Immobilize the bitten limb using a splint or loose bandage.\n"
        "4. Remove rings, bracelets, or tight clothing near the bite area.\n"
        "5. Transport the victim immediately to the nearest medical facility with anti-venom.\n"
        "6. DO NOT cut the bite site, apply a tourniquet, or try to suck out the venom."
    ),
    "first_aid_steps": [
        "Treat the snake as venomous out of an abundance of caution.",
        "Keep the victim calm and reassured to slow venom circulation.",
        "Immobilize the bitten limb using a splint or loose bandage.",
        "Remove rings, bracelets, or tight clothing near the bite area.",
        "Transport the victim immediately to the nearest medical facility with anti-venom."
    ],
    "avoid_actions": [
        "DO NOT cut the bite site.",
        "DO NOT apply a tourniquet.",
        "DO NOT try to suck out the venom."
    ]
}

def get_snake_metadata(species_name: str) -> Dict[str, Any]:
    """
    Retrieves safety and taxonomic metadata for a given snake species by name.
    
    Args:
        species_name: The predicted snake species name.
        
    Returns:
        A dictionary containing common_name, scientific_name, venomous status, description,
        habitat, and first-aid instructions.
    """
    normalized_name = species_name.lower().strip()
    if normalized_name in ("uncertain", "uncertain prediction"):
        return UNCERTAIN_METADATA
    return SNAKE_METADATA.get(normalized_name, DEFAULT_METADATA)
