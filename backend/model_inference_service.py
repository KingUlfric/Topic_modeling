from topic_modeling_service import get_dominant_topic
from preprocessing_service import preprocess_text

def run_model_inference(review_text, n_words=10):
    """
    Run the model inference to find the dominant topic in a review.

    :param review_text: String, the review text.
    :param n_words: Integer, number of words to consider in the dominant topic.
    :return: String, dominant topic of the review.
    """
    preprocessed_text = preprocess_text(review_text)
    dominant_topic = get_dominant_topic(preprocessed_text, n_words)
    return dominant_topic
