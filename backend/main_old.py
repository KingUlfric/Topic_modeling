import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'D:\\clean-athlete-403915-54b311f535e5.json'

from google.cloud import translate_v2 as translate
import pandas as pd

'''
# Initialize the translation client
translate_client = translate.Client()


# Function to translate text using Google Cloud Translation API
def translate_text(text, target_language='en'):
    # Use the client to translate the text
    if text is None or text.strip() == '':
        return None
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']


# Load your dataset
#df = pd.read_csv('F:\\studies\\Text Analytics 2\\Training_data_Google_Play_reviews.csv')
df = pd.read_csv('D:\\Training_data_Google_Play_reviews_6000.csv')


# Add a column for the translated text
df['translated_content'] = df['content'].apply(lambda x: translate_text(x))

# Save the updated dataframe with the translated content
df.to_csv('D:\\translated_dataset.csv', index=False)

'''
'''
###check if translations is done right

# Load the translated dataset
file_path = 'D:\\translated_dataset.csv'
translated_df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to verify the translations
translated_df.head()

# Filter out entries where the userLang is not 'EN'
non_english_reviews = translated_df[translated_df['userLang'] != 'EN']

# Display a sample of non-English entries to check the translations
non_english_reviews.head()

# Function to convert HTML entities to characters
def html_to_text(html):
    from html import unescape
    return unescape(html)

# Apply the function to the translated_content column
translated_df['clean_translated_content'] = translated_df['translated_content'].apply(html_to_text)

# Display the updated DataFrame to verify the changes
translated_df[['content', 'translated_content', 'clean_translated_content']].head(10)
translated_df.to_csv('D:\\translated_dataset1.csv', index=False)

'''
file_path = 'D:\\translated_dataset1.csv'
translated_df = pd.read_csv(file_path)

# Text preproccesing
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Function for text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


translated_df['clean_translated_content'] = translated_df['clean_translated_content'].apply(preprocess_text)

###  topic modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# Function to print the top words for each topic
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


# Number of topics
n_topics = 5
# Number of top words to display for each topic
n_top_words = 10

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf = tf_vectorizer.fit_transform(translated_df['clean_translated_content'])
'''
# Fit the LDA model
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)

# Print the top words for each topic
tf_feature_names = tf_vectorizer.get_feature_names_out()
print_top_words(lda, tf_feature_names, n_top_words)
'''

## Topic Modeling 5, 10, 15 topics
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

# We'll simplify the grid search by using a smaller range of parameters

# Define the range of topics to try
n_topics_options = [5, 10, 15]

# Dictionary to store the scores for each model
models_scores = {}

for n_topics in n_topics_options:
    # Initialize LDA with the current number of topics
    lda_model = LatentDirichletAllocation(n_components=n_topics, learning_decay=0.7, random_state=0)

    # Fit the model
    lda_model.fit(tf)

    # Calculate log-likelihood and perplexity
    log_likelihood = lda_model.score(tf)
    perplexity = lda_model.perplexity(tf)

    # Store the scores
    models_scores[n_topics] = {'log_likelihood': log_likelihood, 'perplexity': perplexity}

    # Print the top words for each model
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    print_top_words(lda_model, tf_feature_names, n_top_words)

    # showing review examples
    import numpy as np

    # Create a DataFrame to hold the topic distribution for each document
    topic_distributions = lda_model.transform(tf)
    topic_df = pd.DataFrame(topic_distributions)

    # Assign each document to the most likely topic
    dominant_topic = np.argmax(topic_df.values, axis=1)
    translated_df['dominant_topic'] = dominant_topic

    # Now, you can group the dataset by 'dominant_topic' and sample a few reviews from each
    for topic_num in range(n_topics):
        print(f"Examples of reviews from topic {topic_num}:\n")
        # Filter the DataFrame for the current topic
        topic_subset = translated_df[translated_df['dominant_topic'] == topic_num]
        # Sample 3 reviews (or however many you want to look at)
        sampled_reviews = topic_subset.sample(n=3)['clean_translated_content']
        for i, review in enumerate(sampled_reviews):
            print(f"Review {i}: {review}\n")
        print("-" * 50)

# Output the scores for each model
for n_topics, scores in models_scores.items():
    print(f"Topics: {n_topics}, Log Likelihood: {scores['log_likelihood']}, Perplexity: {scores['perplexity']}")




