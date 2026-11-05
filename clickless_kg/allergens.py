"""Infer allergens from ingredient text and product name.

Open Food Facts' allergens_tags are sometimes empty even when the ingredient
list plainly says "peanuts" or "milk" (contributors forget to fill them in).
For an allergy-safety system a missed allergen is the worst possible error, so
the catalog uses the UNION of OFF's tags and what this module finds.

The rules are deliberately conservative (prefer a false "contains" over a
missed allergen), with a few guards against obvious false positives:
  * "may contain ..." / "made in a facility ..." clauses are removed first
    (cross-contact warnings are not ingredients);
  * "X-free", "free from X", "no X", "sans X", "sin X" are removed;
  * plant "milks"/"butters" (coconut milk, cocoa butter, peanut butter, ...)
    do not count as dairy; buckwheat is not wheat.
English, French and Spanish keywords are covered (most OFF products in the
catalog are labelled in one of those).
"""
from __future__ import annotations

import re

_PLANT = r"(?<!coconut )(?<!almond )(?<!oat )(?<!soy )(?<!rice )(?<!cashew )(?<!hemp )" \
         r"(?<!coco )(?<!plant )(?<!nut )(?<!pea )(?<!macadamia )(?<!hazelnut )"

PATTERNS: dict[str, list[str]] = {
    "milk": [
        _PLANT + r"\bmilks?\b", r"\bmilk (?:powder|solids|fat|protein)", r"\bwhey\b", r"\bcasein",
        r"\bcaseinate", r"\blactose\b", r"\bbuttermilk\b", r"\bghee\b", r"\bkefir\b", r"\bdairy\b",
        r"(?<!cookie )(?<!n-)(?<!sesame )(?<!peanut )(?<!cocoa )(?<!cacao )(?<!shea )(?<!nut )(?<!almond )(?<!cashew )"
        r"(?<!seed )(?<!apple )(?<!sunflower )(?<!pistachio )(?<!hazelnut )(?<!soy )\bbutter\b",
        r"(?<!coconut )(?<!cashew )(?<!oat )(?<!soy )\bcream\b(?! of tartar)",
        r"\bcheeses?\b", r"\bparmesan\b", r"\bmozzarella\b", r"\bcheddar\b",
        r"(?<!coconut )(?<!soy )(?<!almond )(?<!oat )\byog(?:h)?urts?\b", r"\bskyr\b",
        r"\blait\b(?! de coco)", r"\bbeurre\b(?! de (?:cacao|cacahu|karit|noix|amande|coco))", r"\bfromage", r"\bcr[eè]me\b",
        r"\bleche\b(?! de coco)", r"\bnata\b", r"\bqueso\b", r"\blactos[ée]rum\b",
    ],
    "gluten": [
        r"(?<!buck)wheat", r"\bbarley\b", r"\brye\b", r"\bspelt\b", r"\bsemolina\b", r"\bdurum\b",
        r"\bmalt(?:ed)?\b", r"\bfarro\b", r"\bkamut\b", r"\btriticale\b", r"\bgluten\b", r"\bcouscous\b",
        r"\bbl[ée]\b", r"\bfroment\b", r"\borge\b", r"\bseigle\b", r"\b[ée]peautre\b",
        r"\btrigo\b", r"\bcebada\b", r"\bcenteno\b",
    ],
    "eggs": [r"\beggs?\b", r"\begg (?:white|yolk)s?\b", r"\balbumen\b", r"\bœufs?\b",
             r"\boeufs?\b", r"\bhuevos?\b"],
    "peanuts": [r"\bpeanuts?\b", r"\bgroundnuts?\b", r"\barachides?\b", r"\bcacahu[eè]tes?\b",
                r"\bcacahuetes?\b", r"\bman[ií]\b"],
    "tree_nuts": [r"\balmonds?\b", r"\bcashews?\b", r"\bhazelnuts?\b", r"\bwalnuts?\b",
                  r"\bpecans?\b", r"\bpistachios?\b", r"\bmacadamias?\b", r"\bbrazil nuts?\b",
                  r"\bpine nuts?\b", r"\btree nuts?\b", r"\bamandes?\b", r"\bnoisettes?\b",
                  r"\bnoix\b(?! de coco)", r"\bpistaches?\b", r"\balmendras?\b", r"\bavellanas?\b",
                  r"\bnueces\b", r"\banacardos?\b", r"\bpistachos?\b"],
    "soy": [r"\bsoy", r"\bsoja\b", r"\bsoya\b", r"\bedamame\b", r"\btofu\b", r"\btamari\b",
            r"\bmiso\b"],
    "fish": [r"\bfish\b", r"\banchov", r"\bsalmon\b", r"\btuna\b", r"\bcod\b", r"\bsardines?\b",
             r"\bpoissons?\b", r"\bpescado\b"],
    "shellfish": [r"\bshellfish\b", r"\bshrimps?\b", r"\bprawns?\b", r"\bcrabs?\b",
                  r"\blobsters?\b", r"\bclams?\b", r"\bmussels?\b", r"\boysters?\b",
                  r"\bscallops?\b", r"\bcrevettes?\b", r"\bgambas\b"],
    "sesame": [r"\bsesame\b", r"\bs[ée]same\b", r"\btahini\b", r"\bs[ée]samo\b"],
    "mustard": [r"\bmustard\b", r"\bmoutarde\b", r"\bmostaza\b"],
    "celery": [r"\bcelery\b", r"\bc[ée]leri\b", r"\bceleriac\b"],
    "sulphites": [r"\bsulph?ites?\b", r"\bsulfur dioxide\b", r"\bsulphur dioxide\b",
                  r"\bmetabisulph?ite\b", r"\bbisulph?ite\b"],
    "lupin": [r"\blupin", r"\baltramuz"],
}
_COMPILED = {a: re.compile("|".join(f"(?:{p})" for p in ps)) for a, ps in PATTERNS.items()}

_CROSS_CONTACT = re.compile(
    r"(?:may (?:also )?contain|can contain|might contain|made in a (?:facility|factory)"
    r"|produced in a (?:facility|factory)|manufactured (?:in|on)|processed in"
    r"|made on (?:shared )?(?:equipment|lines?)|packed in a facility|traces? of"
    r"|peut contenir|traces? [ée]ventuelles?|traces? d['e ]|fabriqu[ée] dans|usine qui"
    r"|puede contener|trazas de)[^.;\n]*", re.I)
_NEGATIONS = re.compile(
    r"\b[\w-]+[- ]free\b|\bfree (?:from|of) [\w ,]+?(?=[.;\n]|$)|\bno [\w-]+\b|\bnon[- ]dairy\b"
    r"|\bdairy[- ]free\b|\bsans [\w-]+\b|\bsin [\w-]+\b|\bwithout [\w-]+\b", re.I)


ANIMAL = {"milk", "eggs", "fish", "shellfish"}
# Product names are ambiguous for these ("vegan cream cheese", "flax milk", "queso chips"),
# so they are inferred from the ingredient list only.
INGREDIENTS_ONLY = {"milk", "eggs"}


def clean(text: str) -> str:
    text = (text or "").lower().replace("_", " ")
    text = _CROSS_CONTACT.sub(" ", text)
    text = _NEGATIONS.sub(" ", text)
    return re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)  # cocoa-butter -> cocoa butter


def infer_allergens(ingredients_text: str, name: str = "", vegan: bool = False) -> list[str]:
    """Allergens found in the ingredient list (and, except milk/eggs, the name).
    For products labelled vegan, animal allergens are not inferred: the label is
    a stronger signal than a keyword (OFF's own allergen tags still apply)."""
    ing, nm = clean(ingredients_text), clean(name)
    found = set()
    for a, rx in _COMPILED.items():
        if vegan and a in ANIMAL:
            continue
        if rx.search(ing) or (a not in INGREDIENTS_ONLY and rx.search(nm)):
            found.add(a)
    return sorted(found)
