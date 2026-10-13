from __future__ import annotations


EXCLUDED_LABELS: set[str] = set()

LABEL_NAME_FIX: dict[str, list[str]] = {
    "Urban fabric": ["urban fabric", "urban built-up area"],
    "Industrial or commercial units": ["industrial or commercial area", "industrial facilities"],
    "Arable land": ["arable land", "cropland"],
    "Permanent crops": ["permanent crops", "orchards"],
    "Pastures": ["pastures", "grass pastures"],
    "Complex cultivation patterns": ["complex cultivation patterns", "mixed cropland parcels"],
    "Land principally occupied by agriculture, with significant areas of natural vegetation": [
        "agricultural land with natural vegetation",
        "farmland mixed with natural vegetation",
    ],
    "Agro-forestry areas": ["agro-forestry areas", "agroforestry land"],
    "Broad-leaved forest": ["broad-leaved forest", "deciduous forest"],
    "Coniferous forest": ["coniferous forest", "needleleaf forest"],
    "Mixed forest": ["mixed forest"],
    "Natural grassland and sparsely vegetated areas": [
        "natural grassland",
        "sparsely vegetated land",
    ],
    "Moors, heathland and sclerophyllous vegetation": [
        "heathland",
        "moors",
        "sclerophyllous vegetation",
    ],
    "Transitional woodland, shrub": ["transitional woodland shrub", "shrubland"],
    "Beaches, dunes, sands": ["beaches dunes sands", "sandy beach and dunes"],
    "Inland wetlands": ["inland wetlands", "freshwater wetlands"],
    "Coastal wetlands": ["coastal wetlands", "salt marsh"],
    "Inland waters": ["inland waters", "lakes and rivers"],
    "Marine waters": ["marine waters", "sea water"],
}

TEMPLATES: list[str] = [
    "a satellite image of {}",
    "an aerial photograph of {}",
    "a remote sensing image of {}",
    "a satellite image with {} in it",
    "an aerial photograph containing {}",
    "{} visible in a remote sensing scene",
]

COOCCUR_TEMPLATES: list[str] = [
    "a satellite image of {} with {}",
    "a remote sensing image showing {} near {}",
    "an aerial view of {} alongside {}",
    "a satellite image containing {} and {}",
]
