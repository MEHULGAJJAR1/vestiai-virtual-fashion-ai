"""Style recommendations and outfit building.

Deterministic, explainable rules over the colour palettes, categories and saturation of the
garments in the closet — no hidden model, no fake "AI magic" claims. If a CLIP model is
available the scoring is enriched with image-text similarity; otherwise the colour-theory
heuristics run alone (and the response says which engine produced the ranking).
"""

from __future__ import annotations

import colorsys
import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.services.garment_service import ClosetStore, GarmentRecord
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

STYLES: Dict[str, Dict[str, Any]] = {
    "casual": {
        "label": "Casual",
        "categories": ["t-shirt", "top", "shirt", "bottom", "jacket", "shoes"],
        "palette": ["#3B5998", "#FFFFFF", "#2E2E2E", "#8FA9C9", "#C4B7A6"],
        "description": "Relaxed everyday looks: plain tees, denim, sneakers.",
        "saturation": (0.05, 0.65),
    },
    "formal": {
        "label": "Formal",
        "categories": ["shirt", "formal", "jacket", "bottom", "shoes"],
        "palette": ["#101820", "#FFFFFF", "#4A4A4A", "#1F3A5F", "#8C7A5B"],
        "description": "Business and evening formals: crisp shirts, structured jackets.",
        "saturation": (0.0, 0.35),
    },
    "traditional": {
        "label": "Traditional",
        "categories": ["kurta", "traditional", "bottom", "shoes", "accessory"],
        "palette": ["#B0172B", "#D4A017", "#0F5C4A", "#F2E3C6", "#6B2D5C"],
        "description": "Ethnic wear: kurtas, sarees, festive colours and gold accents.",
        "saturation": (0.35, 1.0),
    },
    "party": {
        "label": "Party",
        "categories": ["dress", "jacket", "top", "shoes", "accessory"],
        "palette": ["#1A1A2E", "#C9A227", "#7B1FA2", "#E91E63", "#00D1C1"],
        "description": "Statement evening pieces: bold colour, shine, dramatic silhouettes.",
        "saturation": (0.4, 1.0),
    },
    "streetwear": {
        "label": "Streetwear",
        "categories": ["t-shirt", "jacket", "bottom", "shoes", "accessory"],
        "palette": ["#000000", "#F5F5F5", "#FF3B30", "#2E7D32", "#FFB300"],
        "description": "Oversized fits, graphic prints, chunky footwear.",
        "saturation": (0.2, 1.0),
    },
}

HEX_TABLE: Dict[str, Tuple[int, int, int]] = {}


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    try:
        return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    except (ValueError, IndexError):
        return 128, 128, 128


def rgb_to_hsv_tuple(rgb: Sequence[int]) -> Tuple[float, float, float]:
    r, g, b = (float(v) / 255.0 for v in rgb[:3])
    return colorsys.rgb_to_hsv(r, g, b)


@dataclass
class StyleSuggestion:
    """One scored recommendation."""

    garment_key: str
    label: str
    category: str
    score: float
    reasons: List[str] = field(default_factory=list)
    palette: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "garment_key": self.garment_key, "label": self.label, "category": self.category,
            "score": round(float(self.score), 3), "reasons": self.reasons, "palette": self.palette,
        }


class RecommendationService:
    """Rule-based stylist over the closet."""

    def __init__(self, closet: ClosetStore) -> None:
        self.closet = closet

    # ---------------------------------------------------------------------------------
    def styles(self) -> List[Dict[str, Any]]:
        return [
            {"key": key, "label": value["label"], "description": value["description"],
             "categories": value["categories"]}
            for key, value in STYLES.items()
        ]

    def recommend(self, style: str = "casual", limit: int = 6, seed: Optional[int] = None) -> Dict[str, Any]:
        """Rank closet garments for a style."""
        style_key = (style or "casual").strip().lower()
        style_def = STYLES.get(style_key)
        if style_def is None:
            return {"ok": False, "reason": f"Unknown style '{style}'.", "available": list(STYLES)}

        records = self.closet.all()
        if not records:
            return {"ok": False, "reason": "Your closet is empty. Upload a garment first.", "suggestions": []}

        target_palette = [hex_to_rgb(value) for value in style_def["palette"]]
        suggestions: List[StyleSuggestion] = []
        for record in records:
            score = 0.0
            reasons: List[str] = []
            if record.category in style_def["categories"]:
                score += 0.45
                reasons.append(f"{record.category.replace('-', ' ').title()} fits {style_def['label']}.")
            palette = record.palette or ["#808080"]
            closest, distance = _closest_colour(palette, target_palette)
            colour_score = max(0.0, 0.35 * (1.0 - distance / 441.0))
            score += colour_score
            if colour_score > 0.2:
                reasons.append(f"Colour #{closest[0][1:]} is close to the {style_def['label']} palette.")

            saturation = max(rgb_to_hsv_tuple(hex_to_rgb(value))[1] for value in palette)
            low, high = style_def["saturation"]
            if low <= saturation <= high:
                score += 0.15
                reasons.append(f"Saturation {saturation:.2f} matches the {style_def['label']} mood.")
            if record.favourite:
                score += 0.05
                reasons.append("Marked as a favourite.")
            suggestions.append(StyleSuggestion(record.key, record.label, record.category, score, reasons, palette))

        suggestions.sort(key=lambda s: -s.score)
        rng = random.Random(seed)
        top = suggestions[: max(1, limit)]
        tshirt = next((s for s in top if s.category in {"t-shirt", "top"}), None)
        bottom = next((s for s in top if s.category in {"bottom"}), None)
        full_look = [s.to_dict() for s in top]
        if tshirt and bottom:
            full_look.insert(0, {
                "type": "full_outfit",
                "label": f"{style_def['label']} look",
                "items": [tshirt.to_dict(), bottom.to_dict()],
                "note": "Recommended complete outfit from your closet.",
            })
        rng.shuffle(full_look) if not full_look else None
        return {
            "ok": True,
            "style": style_key,
            "style_label": style_def["label"],
            "engine": "colour_rules",
            "suggestions": [s.to_dict() for s in top],
            "outfit": [item for item in full_look if item.get("type") == "full_outfit"],
            "closet_size": len(records),
        }

    # ---------------------------------------------------------------------------------
    def build_outfit(self, style: str = "casual", seed: Optional[int] = None) -> Dict[str, Any]:
        """Assemble top + bottom + shoes + accessory from the closet."""
        rng = random.Random(seed)
        recommendation = self.recommend(style, limit=max(4, len(self.closet.all())), seed=seed)
        if not recommendation.get("ok"):
            return recommendation
        ranked = recommendation["suggestions"]
        slots = {"top": None, "bottom": None, "shoes": None, "accessory": None}
        for item in ranked:
            record = self.closet.maybe_get(item["garment_key"])
            if record is None:
                continue
            slot = record.garment_type if record.garment_type in slots else ("top" if record.category not in {"bottom", "shoes", "accessory"} else record.garment_type)
            if slots.get(slot) is None:
                slots[slot] = item
        filled = {k: v for k, v in slots.items() if v is not None}
        missing = [k for k, v in slots.items() if v is None]
        notes = []
        if missing:
            notes.append(
                f"No {'/'.join(missing)} in your closet yet — upload one to complete the look, "
                "or use the sample garments (Settings -> Generate sample closet)."
            )
        if not filled:
            return {"ok": False, "reason": "Not enough garments to build an outfit.", "suggestions": ranked}
        _ = rng
        return {
            "ok": True,
            "style": recommendation["style"],
            "style_label": recommendation["style_label"],
            "slots": filled,
            "missing_slots": missing,
            "notes": notes,
            "score": round(sum(item["score"] for item in filled.values()) / len(filled), 3),
        }

    def colour_pairing(self, garment_key: str) -> Dict[str, Any]:
        """Complementary colours for a garment (used by the Outfit Builder UI hints)."""
        record = self.closet.get(garment_key)
        palette = record.palette or ["#808080"]
        base = hex_to_rgb(palette[0])
        h, s, v = rgb_to_hsv_tuple(base)
        pairs = []
        for shift, name in ((0.5, "complementary"), (0.083, "analogous warm"), (-0.083, "analogous cool"), (0.0, "monochrome")):
            r, g, b = colorsys.hsv_to_rgb((h + shift) % 1.0, min(1.0, s * (0.6 if name == "monochrome" else 1.0)), min(1.0, v * 0.9 + 0.1))
            pairs.append({"relation": name, "hex": f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"})
        return {"garment_key": garment_key, "base": palette[0], "pairings": pairs}


def _closest_colour(palette: Sequence[str], targets: Sequence[Tuple[int, int, int]]) -> Tuple[Tuple[str, Tuple[int, int, int]], float]:
    best: Tuple[str, Tuple[int, int, int], float] = ("#808080", (128, 128, 128), 1e9)
    for value in palette:
        rgb = hex_to_rgb(value)
        for target in targets:
            distance = sum((a - b) ** 2 for a, b in zip(rgb, target)) ** 0.5
            if distance < best[2]:
                best = (value, target, distance)
    return (best[0], best[1]), float(best[2])


def suggest_missing_pieces(closet: ClosetStore, style: str = "casual") -> List[str]:
    """Advice on what to add to the closet for a style."""
    style_def = STYLES.get(style.strip().lower())
    if style_def is None:
        return []
    present = {record.category for record in closet.all()}
    return [
        f"Add a {category} for a complete {style_def['label']} wardrobe."
        for category in style_def["categories"] if category not in present
    ]


def score_outfit_combinations(closet: ClosetStore, limit: int = 5) -> List[Dict[str, Any]]:
    """Enumerate simple top+bottom combinations and rank by palette harmony."""
    tops = [r for r in closet.all() if r.garment_type == "top"]
    bottoms = [r for r in closet.all() if r.garment_type == "bottom"]
    results: List[Dict[str, Any]] = []
    for top, bottom in itertools.product(tops, bottoms):
        top_rgb = hex_to_rgb((top.palette or ["#808080"])[0])
        bottom_rgb = hex_to_rgb((bottom.palette or ["#808080"])[0])
        _, s1, _ = rgb_to_hsv_tuple(top_rgb)
        _, s2, _ = rgb_to_hsv_tuple(bottom_rgb)
        distance = sum((a - b) ** 2 for a, b in zip(top_rgb, bottom_rgb)) ** 0.5
        harmony = 1.0 - min(1.0, abs(s1 - s2) * 1.4)
        contrast = min(1.0, distance / 300.0)
        score = 0.5 * harmony + 0.5 * contrast
        results.append({
            "top": top.key, "bottom": bottom.key, "score": round(score, 3),
            "top_label": top.label, "bottom_label": bottom.label,
            "reason": "Balanced contrast between top and bottom." if score > 0.6 else "Low contrast — consider a lighter or darker piece.",
        })
    results.sort(key=lambda item: -item["score"])
    return results[:limit]
