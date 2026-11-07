import re
import emoji
import unicodedata


def clean_text(text):

    if not isinstance(text, str):
        return ""

    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Convert emojis into descriptive text
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Remove usernames
    text = re.sub(r"@\w+", "", text)

    # Normalize hashtags while retaining their words
    text = text.replace("#", "")

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()
