                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               # test_topics.py

from preprocessing_service import preprocess_text
from topic_modeling_service import train_and_load_lda_model
import pandas as pd

# Load your dataset
df = pd.read_csv('D:\\translated_dataset1.csv')  # Replace with your dataset path
df['processed_text'] = df['clean_translated_content'].apply(preprocess_text)

# Experiment configurations
n_topics_options = [5]
ngram_ranges = [(1, 1)]
max_df_options = [0.95]  # Adjust max document frequency
min_df_options = [2]  # Adjust min document frequency
learning_decay_options = [1]  # Adjust learning rate decay


def print_top_words_with_examples(model, feature_names, texts, n_top_words=10, n_examples=15):
    for topic_idx, topic in enumerate(model.components_):
        message = f"Topic #{topic_idx}: "
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)

        # Find example reviews for this topic
        topic_reviews = []
        for i, doc_topic in enumerate(model.transform(vectorizer.transform(texts))):
            if doc_topic.argmax() == topic_idx and len(topic_reviews) < n_examples:
                if len(texts[i]) > 0:
                    topic_reviews.append(texts[i])

        print(f"Examples ({n_examples}):")
        for review in topic_reviews:
            print(f"- {review}")
        print()

# Store results
experiment_results = []

# Run experiments
for n_topics in n_topics_options:
    for ngram_range in ngram_ranges:
        for max_df in max_df_options:
            for min_df in min_df_options:
                for learning_decay in learning_decay_options:
                    print(f"\n\n===== Experimenting with {n_topics} topics, N-gram range {ngram_range}, max_df={max_df}, min_df={min_df}, learning_decay={learning_decay} =====")
                    lda_model, vectorizer = train_and_load_lda_model(df['processed_text'].tolist(), n_topics=n_topics,
                                                                     max_df=max_df, min_df=min_df)

                    # Calculate a log likelihood and perplexity
                    term_matrix = vectorizer.transform(df['processed_text'].tolist())
                    log_likelihood = lda_model.score(term_matrix)
                    perplexity = lda_model.perplexity(term_matrix)
                    print(f"Log Likelihood: {log_likelihood}, Perplexity: {perplexity}\n")
                    print_top_words_with_examples(lda_model, vectorizer.get_feature_names_out(), df['processed_text'].tolist())
                    experiment_results.append({
                        'n_topics': n_topics,
                        'ngram_range': ngram_range,
                        'log_likelihood': log_likelihood,
                        'perplexity': perplexity,
                        'max_df_options': max_df_options,
                        'min_df_options': min_df_options,
                        'learning_decay_options': learning_decay_options
                    })

# Finding the best model
#best_model = max(experiment_results, key=lambda x: (x['log_likelihood'], -x['perplexity']))

#print(f"Best Model: {best_model}")