"""Rule-based weak labels for Kazakh profanity.

STRONG patterns give label 1 directly.
WEAK patterns are ambiguous; texts that match only WEAK patterns are checked by hand
(see data/manual_labels.csv).
"""
import re

# Latin and digit look-alikes that people use to hide words
_LOOKALIKE = str.maketrans({
    "a": "а", "o": "о", "e": "е", "k": "к", "x": "х", "y": "у", "p": "р",
    "c": "с", "h": "н", "b": "в", "m": "м", "t": "т", "i": "і",
    "0": "о", "@": "а", "3": "з",
})


def normalize_for_rules(text: str) -> str:
    t = str(text).lower().translate(_LOOKALIKE)
    t = re.sub(r"(?<=\w)[*#.\-_](?=\w)", "", t)  # "к*тақ", "с.у.к.а" -> joined
    t = re.sub(r"(\w)\1{2,}", r"\1\1", t)         # "нахуууй" -> "нахууй"
    return t


B = r"(?<!\w)"  # start of word
E = r"(?!\w)"   # end of word

STRONG = {
    # Kazakh obscene words
    "qotaq":   r"[қк][оө]т[аә][қкғг]",
    "sik":     B + r"(с[іи]г(ейін|ейин|ейн|еин|еын|ейім|ейм|ем|ема|ін|ин|іл|ил|іс|ісі|іп|ілгір|ілің)"
               r"|(сыг|сг)(ейін|ейн|еин|еын|ем|ин|ін)" + E +
               r"|с[іиы]кт[іиы]р|с[іи]кт[іи]м?|с[іи]ккен)",
    "amy":     B + r"ам(ы|ың|ын|ыңды|ынды|ыңа|дар|дарың|дарын|ның|бас|шық|шык|щық|щелек|жалағыш|жалагыш)" + E,
    "shesh_am": r"шеш\w*\s+ам",
    "shesh_sik": r"шеш\w*\s*(с[іиы]?г|сікт|сикт)",
    "kot":     B + r"(к[өо]т(ің|ін|ын|іңді|ыңды|инды|ынды|іңе|ыңа|ине|іне|ке|іңнен)|көт(ті|ски|)|жұнкөт)" + E,
    "shshs":   B + r"([шщ][шщ]с+|667(?!\w))",
    "kazak_other": B + r"([қк]аншы[қк]|ж[әа]л[еә]п|салда[қк]ы|т[аә]с[аә][қк]|с[аә]й[қк]ал|амшы[қк]|амщы[қк])",
    # Russian mat used inside Kazakh comments (code-switching)
    "ru_hui":  r"ху[йияеёю]|" + B + r"(нах(у|уй)?|нахх|пох(у|уй))" + E,
    "ru_pizd": r"п[иіе]?зд",
    "ru_eb":   B + r"(за[еэйи]б|[еэй]б[ау]т|еба[йиі]т|[еэ]б[ау]н|д[ао]лб[ао][еэйёи]|е[бп]\s*тв|ёб|е?еба(?!\w)|ебо[йи])|" + B + r"(ёп\w*|епта|йоп\w*|йеп\w*)" + E,
    "ru_blya": B + r"(бля|блә|блят|бляд)",
    "ru_other": B + r"(сук(а|и|ам|алар|ар)" + E + r"|муда[кч]|гандон|пид[оа]р|пидар|педик(?!юр)|шлюх|залуп|г[ао]вн|жоп[аеуы])",
}

WEAK = {
    "am_bare":  B + r"ам" + E,
    "shesh":    B + r"шеш[еа][ңн]",
    "bok":      B + r"б[оө][кқ]" + E,
    "zb":       B + r"(зб|нау)" + E,
    "sik_weak": B + r"(с[іиы][гк]|сг)\w*",
    "masked":   r"\w\*+\w|\*{2,}",
}

_STRONG_RE = {k: re.compile(v) for k, v in STRONG.items()}
_WEAK_RE = {k: re.compile(v) for k, v in WEAK.items()}


def rule_hits(text: str):
    t = normalize_for_rules(text)
    strong = [k for k, r in _STRONG_RE.items() if r.search(t)]
    weak = [k for k, r in _WEAK_RE.items() if r.search(t)]
    return strong, weak
