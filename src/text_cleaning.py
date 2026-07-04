"""Nettoyage de texte pour les tweets (Partie 1 du sujet)."""
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_SYMBOL_RE = re.compile(r"#")
SPECIAL_CHARS_RE = re.compile(r"[^a-z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str, lemmatize: bool = True) -> str:
    """Applique la chaîne de nettoyage demandée par le sujet :
    minuscules -> suppression URLs -> suppression mentions/hashtags ->
    suppression caractères spéciaux -> suppression stop words -> lemmatisation.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_SYMBOL_RE.sub(" ", text)
    text = SPECIAL_CHARS_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()

    tokens = [t for t in text.split() if t not in _STOPWORDS and len(t) > 1]

    if lemmatize:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)
