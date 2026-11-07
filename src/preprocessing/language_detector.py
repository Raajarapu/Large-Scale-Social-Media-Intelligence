from langdetect import detect, LangDetectException


def detect_language(text):

    if not isinstance(text, str) or len(text.strip()) < 10:
        return "unknown"

    try:
        return detect(text)

    except LangDetectException:
        return "unknown"
