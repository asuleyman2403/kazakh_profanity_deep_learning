"""Small text helpers shared by data preparation, training and inference."""
import re

_KEYCAP = re.compile(r"([0-9])️?⃣")  # "7️⃣" -> "7"
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_EMAIL = re.compile(r"\S+@\S+\.\w+")
_USER = re.compile(r"@\w+")
_PHONE = re.compile(r"\+?\d[\d\s\-()]{6,}\d")
# surnames like "Нурмұханова", "Ахметов", "Сұлтанұлы" (first names can stay, see README)
_SURNAME = re.compile(r"(?<!\w)[А-ЯӘІҢҒҮҰҚӨҺ][а-яәіңғүұқөһ]{2,}(ова|ева|ов|ев|ұлы|улы|қызы|кызы)(?!\w)")


def mask_pii(text: str) -> str:
    """Hide links, e-mails, @usernames and phone numbers."""
    t = _KEYCAP.sub(r"\1", str(text))
    t = _URL.sub("[URL]", t)
    t = _EMAIL.sub("[EMAIL]", t)
    t = _USER.sub("[USER]", t)
    t = _PHONE.sub("[PHONE]", t)
    t = _SURNAME.sub("[NAME]", t)
    return re.sub(r"\s+", " ", t).strip()


def model_normalize(text: str) -> str:
    """Light normalization before the text goes into the model."""
    t = str(text).lower()
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)  # "нахуууй" -> "нахуй"-like, keeps at most 2 repeats
    return re.sub(r"\s+", " ", t).strip()


def near_dup_key(text: str) -> str:
    """Key for near-duplicates: letters only, repeated letters collapsed."""
    t = re.sub(r"[^\w]|[\d_]", "", str(text).lower())
    return re.sub(r"(.)\1+", r"\1", t)
