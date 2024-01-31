import re
import spacy
from html import unescape
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load SpaCy model for lemmatization (make sure to install it and download the language model)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


def html_to_text(html):
    """
    Convert HTML entities to characters.
    :param html: String, the HTML content to be converted.
    :return: String, the converted text.
    """
    return unescape(html)


def preprocess_text(text):
    """
    Preprocess the text by converting to lowercase, removing HTML tags,
    punctuation, numbers, stop words, and extra whitespace, and applying lemmatization.
    :param text: String, the text to preprocess.
    :return: String, the preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Lemmatization
    doc = nlp(text)
    lemmatized = [token.lemma_ for token in doc]

    # Enhanced stop words list
    custom_stop_words = ENGLISH_STOP_WORDS.union({'update', 'message', 'messenger', 'telegram', 'whatsapp', 'viber',
                                                  'nice'}).difference({'not', 'nor', 'no'})  # Keeping negations

    '''custom_stop_words = ENGLISH_STOP_WORDS.union({
        'use', 'like', 'want', 'try', 'make', 'time', 'new', 'just', 'really',
        'lot', 'need', 'work', 'problem', 'update', 'message', 'messenger', 'send', 'open',
        'delete', 'beautiful', 'great', 'good', 'bad', 'nice', 'telegram', 'whatsapp', 'viber',
        'app', 'application', 'version', 'feature', 'option', 'thing', 'people', 'user',
        'way', 'day', 'bit', 'type', 'part', 'point', 'case', 'kind', 'place', 'reason', 'area',
        'side', 'form', 'piece', 'number', 'group', 'level', 'order', 'line', 'name', 'word',
        'fact', 'month', 'year', 'hour', 'minute', 'second', 'change', 'add', 'create',
        'receive', 'give', 'keep', 'put', 'set', 'take', 'tell', 'call', 'find', 'get', 'know',
        'look', 'feel', 'come', 'go', 'leave', 'move', 'play', 'run', 'show', 'stay', 'wait',
        'ask', 'be', 'do', 'have', 'say', 'see', 'I', 'snapchat', 'wechat', 'phone', 'mobile',
        'device', 'system', 'network', 'service', 'account', 'chat', 'contact', 'conversation',
        'text', 'notification', 'alert', 'reminder', 'status', 'upgrade', 'download', 'upload',
        'install', 'uninstall', 'login', 'logout', 'sign', 'register', 'block', 'unblock',
        'disable', 'enable', 'activate', 'deactivate', 'version', 'big', 'small', 'large',
        'short', 'old', 'young', 'early', 'late', 'hard', 'easy', 'fast', 'slow', 'hot', 'cold',
        'warm', 'cool', 'full', 'empty', 'rich', 'poor', 'heavy', 'light', 'strong', 'weak',
        'right', 'wrong', 'high', 'low', 'bright', 'dark', 'personal', 'pronoun', 'me', 'my',
        'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself',
        'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'we', 'us', 'our', 'ours',
        'ourselves', 'they', 'them', 'their', 'theirs', 'themselves'
    }).difference({'not', 'nor', 'no'})  # Keeping negations'''

    text = ' '.join([word for word in lemmatized if word not in custom_stop_words])

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
