"""
utils/classifier.py — Rule-based question type and difficulty classifier.

Classifies Polish ESG directive questions into structured metadata without
any extra LLM calls — pure regex heuristics.

  question_type: factual | scope | process | compliance | comparative
  difficulty:    easy | medium | hard

Used to tag generated samples for:
  - Curriculum learning (train easy → hard)
  - Dataset analysis and datacard statistics
  - Downstream filtering by dataset purchasers
"""

from __future__ import annotations

import re


# ── Type rules (evaluated in order; first match wins) ───────────────────────
# Each entry: (type_name, list_of_regex_patterns_in_Polish)

_TYPE_RULES: list[tuple[str, list[str]]] = [
    ("comparative", [
        r"\bporówna[jć]\b",
        r"\bróżni[acę]\b",
        r"\bróżnica\b",
        r"\bróżni się\b",
        r"\bmiędzy\b.{1,40}\ba\b",
        r"\ba\b.{1,20}\b(CSRD|SFDR|Taksonomia|CSDDD|rozporządzenie)\b",
        r"\bw porównaniu\b",
        r"\bjak.{1,10}(odnosi się|ma się)\b.{1,30}\b(do|wobec)\b",
        r"\brelacj[aei] między\b",
    ]),
    ("scope", [
        r"\bkto\b",
        r"\bkogo\b",
        r"\bjakie (podmioty|spółki|przedsiębiorstwa|jednostki|firmy)\b",
        r"\bkto (jest|są) (objęt|zobowiązan|zobligovan)\b",
        r"\bzakres podmiotowy\b",
        r"\bwyłącze(ni[ae]|ny|nia)\b",
        r"\bkto nie podlega\b",
        r"\bkto jest zwolnion\b",
        r"\bobjęt[aey] (obowiązkiem|przepisami|dyrektywą)\b",
        r"\bkwalifikuje się\b",
        r"\bpróg\b.{1,30}\bpracownik\b",
        r"\bpróg\b.{1,30}\bprzychodów\b",
    ]),
    ("process", [
        r"\bjak (należy|trzeba|powinno|się)\b",
        r"\bprocedura\b",
        r"\bkroki\b",
        r"\betapy\b",
        r"\btermin\b.{1,30}\b(złożenia|przekazania|ujawnienia|raportowania)\b",
        r"\bdo kiedy\b",
        r"\bw jaki sposób\b",
        r"\bw jakim terminie\b",
        r"\bwnioskować\b",
        r"\bsporządzić\b",
        r"\bprzedstawić\b",
        r"\bjak (przygotować|opracować|sporządzić)\b",
    ]),
    ("compliance", [
        r"\bobowiązek\b",
        r"\bwymóg\b",
        r"\bwymaga\b",
        r"\bco (trzeba|musi|należy|powinno)\b",
        r"\braportowa[ćć]\b",
        r"\bujawnić\b",
        r"\bujawnienie\b",
        r"\bzgodność\b",
        r"\bzgodnie z\b",
        r"\bspełni[ćc]\b.{1,20}\bkryteria\b",
        r"\bco (zawiera|musi zawierać)\b.{1,20}\braport\b",
    ]),
    # "factual" is the catch-all default — no patterns needed
    ("factual", []),
]

# ── Difficulty patterns ───────────────────────────────────────────────────────

_HARD_PATTERNS: list[str] = [
    r"\bporówna[jć]\b",
    r"\bróżni[acę]\b",
    r"\bsyntezu[ij]\b",
    r"\brelacja między\b",
    r"\bwzajemn[aey]\b",
    r"\bkombinacj[aei]\b",
    r"\bjednocześnie\b.{1,40}\b(CSRD|SFDR|Taksonomia)\b",
    r"\bimplikacj[aei]\b",
    r"\bjak (łączy|integruje|koreluje)\b",
    r"\bwieloaspektow\b",
    r"\bkonflik[ct]\b.{1,20}\bprzepis\b",
    r"\bspójno[śs][ćc]\b.{1,30}\b(CSRD|SFDR|Taksonomia)\b",
]

_EASY_PATTERNS: list[str] = [
    r"\bco (to jest|oznacza|to)\b",
    r"\bjak (definiuje|jest definiowany|jest zdefiniowany)\b",
    r"\bjaka jest definicja\b",
    r"\bw (którym|jakim) roku\b",
    r"\bkiedy wchodzi w życie\b",
    r"\bjakie są progi\b",
    r"\bjak brzmi\b",
    r"\bco to znaczy\b",
    r"\bjakie są cele\b",
]


def classify_question(text: str) -> tuple[str, str]:
    """
    Returns (question_type, difficulty) for a Polish question string.

    question_type: "factual" | "scope" | "process" | "compliance" | "comparative"
    difficulty:    "easy" | "medium" | "hard"

    Examples
    --------
    >>> classify_question("Kto jest objęty obowiązkiem raportowania ESG?")
    ('scope', 'medium')
    >>> classify_question("Co to jest CSRD?")
    ('factual', 'easy')
    >>> classify_question("Jak CSRD różni się od SFDR w zakresie ujawnień?")
    ('comparative', 'hard')
    """
    lower = text.lower()

    # ── Question type ────────────────────────────────────────────────────────
    q_type = "factual"
    for type_name, patterns in _TYPE_RULES:
        if patterns and any(re.search(p, lower) for p in patterns):
            q_type = type_name
            break

    # ── Difficulty ───────────────────────────────────────────────────────────
    hard_hits = sum(1 for p in _HARD_PATTERNS if re.search(p, lower))
    easy_hits = sum(1 for p in _EASY_PATTERNS if re.search(p, lower))

    if hard_hits >= 2 or (hard_hits >= 1 and len(text) > 140):
        difficulty = "hard"
    elif easy_hits >= 1 and hard_hits == 0:
        difficulty = "easy"
    else:
        difficulty = "medium"

    return q_type, difficulty
