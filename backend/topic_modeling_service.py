import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Global variable to store the LDA model
lda_model = None
vectorizer = None


def train_lda_model(texts, n_topics=5, max_df=0.95, min_df=2, learning_decay=1.0):
    global lda_model, vectorizer
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words='english', ngram_range=(1, 1))
    term_matrix = vectorizer.fit_transform(texts)

    lda_model = LatentDirichletAllocation(n_components=n_topics, learning_decay=learning_decay, random_state=0)
    lda_model.fit(term_matrix)

    return lda_model, vectorizer


def get_dominant_topic(text, n_words=10):
    """
    Identify the most dominant topic for a given text.

    :param text: Text to analyze.
    :param n_words: Number of words to include from the dominant topic.
    :return: String representing the dominant topic.
    """
    global lda_model, vectorizer

    # Ensure the model is loaded
    if lda_model is None or vectorizer is None:
        raise ValueError("LDA model is not loaded.")

    # Vectorize the text using the trained vectorizer
    tf = vectorizer.transform([text])
    # Get the topic distribution
    topic_distribution = lda_model.transform(tf)[0]
    # Identify the dominant topic
    dominant_topic_idx = np.argmax(topic_distribution)

    ''''
    # Get top words in the dominant topic
    top_features_ind = lda_model.components_[dominant_topic_idx].argsort()[:-n_words - 1:-1]
    top_features = [vectorizer.get_feature_names_out()[i] for i in top_features_ind]
    return " ".join(top_features)
    '''
    # Map the dominant topic index to its interpretation
    topic_interpretations = {
        0: "User Experience and App Functionality",
        1: "App Updates and Features",
        2: "Account Management and Security",
        3: "App Registration and Usability",
        4: "General Appraisal and Features"
    }

    return topic_interpretations.get(dominant_topic_idx, "Unknown Topic")


# Function to print the top words for each topic
def print_topics(n_top_words=10):
    global lda_model, vectorizer

    if lda_model is None or vectorizer is None:
        raise ValueError("LDA model is not loaded.")

    topic_descriptions = []
    for topic_idx, topic in enumerate(lda_model.components_):
        message = f"Topic #{topic_idx}: "
        message += " ".join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        topic_descriptions.append(message)

    return topic_descriptions