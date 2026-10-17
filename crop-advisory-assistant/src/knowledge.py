"""Domain knowledge shared by the data pipeline: crops, diseases, climate rules.

IMPORTANT (for your report): the thresholds in CLIMATE_RULES are *approximate starting
points* written from general plant-pathology knowledge. Before you use them, check each
rule against the epidemiology literature / extension sources, adjust the numbers, and
replace `reference` with a real citation. The synthetic climate-conditioned data is only
as trustworthy as these rules.
"""
import re

# crop -> disease -> aliases. Aliases of <=4 letters (acronyms) are matched case-sensitively
# in UPPER case only (e.g. "MLN") to avoid accidental matches.
DISEASES = {
    "maize": {
        "gray leaf spot": ["gray leaf spot", "grey leaf spot", "GLS", "Cercospora zeae-maydis"],
        "northern corn leaf blight": ["northern corn leaf blight", "northern leaf blight", "Exserohilum turcicum", "NCLB"],
        "maize lethal necrosis": ["maize lethal necrosis", "MLN"],
        "maize streak virus": ["maize streak", "MSV"],
    },
    "beans": {
        "bean rust": ["bean rust", "Uromyces appendiculatus"],
        "angular leaf spot": ["angular leaf spot", "Pseudocercospora griseola"],
        "anthracnose": ["bean anthracnose", "anthracnose", "Colletotrichum lindemuthianum"],
        "common bacterial blight": ["common bacterial blight", "CBB", "Xanthomonas axonopodis pv. phaseoli"],
        "bean common mosaic virus": ["bean common mosaic", "BCMV"],
        "root rot": ["root rot", "Fusarium root rot", "Pythium"],
    },
    "cassava": {
        "cassava mosaic disease": ["cassava mosaic", "CMD"],
        "cassava brown streak disease": ["cassava brown streak", "CBSD"],
        "cassava bacterial blight": ["cassava bacterial blight", "CBB", "Xanthomonas axonopodis pv. manihotis"],
        "cassava anthracnose": ["cassava anthracnose"],
    },
    "potato": {
        "late blight": ["late blight", "Phytophthora infestans"],
        "early blight": ["early blight", "Alternaria solani"],
        "bacterial wilt": ["bacterial wilt", "Ralstonia solanacearum", "brown rot of potato"],
    },
}

CROP_ALIASES = {
    "maize": ["maize", "corn"],
    "beans": ["bean", "beans", "common bean", "Phaseolus vulgaris"],
    "cassava": ["cassava", "manioc"],
    "potato": ["potato", "potatoes", "Solanum tuberosum"],
}

CROP_DISPLAY = {"maize": "maize", "beans": "bean", "cassava": "cassava", "potato": "potato"}

# Climate rules. temp = (min_C, max_C) range that favours the disease; humidity_min in %;
# rain_7d_min in mm over the previous 7 days. Missing key = condition not used.
# Vector-borne viruses (cassava mosaic, maize streak, ...) are deliberately NOT included:
# their risk depends on insect-vector populations, so a simple weather rule would be misleading.
CLIMATE_RULES = {
    ("potato", "late blight"): dict(temp=(10, 24), humidity_min=85, rain_7d_min=20,
        mechanism="Late blight develops fastest when leaves stay wet for long periods in cool, humid weather."),
    ("potato", "early blight"): dict(temp=(20, 30), humidity_min=70, rain_7d_min=10,
        mechanism="Early blight is favoured by warm weather with alternating wet and dry spells, especially on stressed plants."),
    ("potato", "bacterial wilt"): dict(temp=(20, 35), rain_7d_min=30,
        mechanism="Bacterial wilt is favoured by warm temperatures and wet soils."),
    ("maize", "gray leaf spot"): dict(temp=(22, 30), humidity_min=85, rain_7d_min=20,
        mechanism="Gray leaf spot is favoured by warm, humid weather with long periods of leaf wetness."),
    ("maize", "northern corn leaf blight"): dict(temp=(18, 27), humidity_min=85, rain_7d_min=15,
        mechanism="Northern corn leaf blight is favoured by moderate temperatures, heavy dew and long periods of leaf wetness."),
    ("beans", "bean rust"): dict(temp=(16, 27), humidity_min=85, rain_7d_min=10,
        mechanism="Bean rust spreads when leaves stay wet from dew or rain in moderate temperatures."),
    ("beans", "angular leaf spot"): dict(temp=(16, 28), humidity_min=80, rain_7d_min=15,
        mechanism="Angular leaf spot is spread by rain splash and favoured by warm, humid weather."),
    ("beans", "anthracnose"): dict(temp=(13, 26), humidity_min=85, rain_7d_min=20,
        mechanism="Bean anthracnose is favoured by cool to moderate, wet weather, when rain splash spreads the spores."),
    ("beans", "common bacterial blight"): dict(temp=(25, 32), humidity_min=80, rain_7d_min=15,
        mechanism="Common bacterial blight is favoured by warm, humid weather and spreads in rain splash."),
    ("cassava", "cassava bacterial blight"): dict(temp=(20, 30), humidity_min=80, rain_7d_min=30,
        mechanism="Cassava bacterial blight is favoured by wet weather, high humidity and rain splash."),
}
for _r in CLIMATE_RULES.values():
    _r["reference"] = "VERIFY: replace with a citation before using in the report"


def _compile(alias):
    if alias.isalpha() and alias.isupper() and len(alias) <= 5:  # acronym: case-sensitive
        return re.compile(rf"\b{re.escape(alias)}\b")
    return re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)


_DISEASE_PATTERNS = {(c, d): [_compile(a) for a in al] for c, ds in DISEASES.items() for d, al in ds.items()}
_CROP_PATTERNS = {c: [_compile(a) for a in al] for c, al in CROP_ALIASES.items()}


def detect(text, manifest_crop=None):
    """Return (crop, [(crop, disease), ...]) for a piece of text."""
    crop_counts = {c: sum(len(p.findall(text)) for p in pats) for c, pats in _CROP_PATTERNS.items()}
    pairs = []
    for (crop, disease), pats in _DISEASE_PATTERNS.items():
        if any(p.search(text) for p in pats) and (crop_counts[crop] > 0 or manifest_crop == crop):
            pairs.append((crop, disease))
    if pairs:
        tally = {}
        for c, _ in pairs:
            tally[c] = tally.get(c, 0) + 1
        crop = max(tally, key=lambda c: (tally[c], crop_counts[c]))
    elif manifest_crop in CROP_ALIASES:
        crop = manifest_crop
    elif any(crop_counts.values()):
        crop = max(crop_counts, key=crop_counts.get)
    else:
        crop = "general"
    return crop, pairs


def assess(crop, disease, climate):
    """Rule-based risk assessment. Returns dict(risk, met, unmet) or None if no rule."""
    rule = CLIMATE_RULES.get((crop, disease))
    if rule is None:
        return None
    met, unmet = [], []
    checks = []
    if "temp" in rule:
        lo, hi = rule["temp"]
        checks.append(("temperature", lo <= climate["temperature_c"] <= hi))
    if "humidity_min" in rule:
        checks.append(("humidity", climate["humidity_pct"] >= rule["humidity_min"]))
    if "rain_7d_min" in rule:
        checks.append(("rainfall", climate["rainfall_mm_7d"] >= rule["rain_7d_min"]))
    for name, ok in checks:
        (met if ok else unmet).append(name)
    n, m = len(checks), len(met)
    risk = "high" if m == n else "moderate" if m == n - 1 else "low"
    return {"risk": risk, "met": met, "unmet": unmet}
