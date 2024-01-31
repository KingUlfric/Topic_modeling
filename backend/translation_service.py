import os
from google.cloud import translate_v2 as translate

# Set the environment variable for Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'D:\\clean-athlete-403915-54b311f535e5.json'

# Initialize the translation client
translate_client = translate.Client()


def translate_text(text, target_language='en'):
    """
    Translate text using Google Cloud Translation API.

    :param text: String, the text to be translated.
    :param target_language: String, the target language code (default is English).
    :return: String, the translated text.
    """
    if text is None or text.strip() == '':
        return None
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']
