from __future__ import annotations


EXCLUDED_LABELS: set[str] = set()

LABEL_NAME_FIX: dict[str, list[str]] = {
    "Grassland": ["grassland"],
    "Factory Building": ["factory building", "industrial building"],
    "Greenhouse": ["greenhouse", "agricultural greenhouse"],
    "Solid Waste": ["solid waste site", "waste dumping area"],
    "Cropland": ["cropland", "farmland"],
    "Road": ["road"],
    "Park": ["park", "green park"],
    "River": ["river"],
    "Lake/Pond": ["lake", "pond"],
    "Woodland": ["woodland", "forest"],
    "Basketball Court": ["basketball court"],
    "Bare Land": ["bare land", "bare soil"],
    "Dust-proof Net": ["dust-proof net", "construction dust screen"],
    "Plastic Mulch": ["plastic mulch", "mulched farmland"],
    "Photovoltaic": ["photovoltaic station", "solar panels"],
    "Railway": ["railway", "railroad"],
    "Football Field": ["football field", "soccer field"],
    "Rural Village": ["rural village", "village houses"],
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
