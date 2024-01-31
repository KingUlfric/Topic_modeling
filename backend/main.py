from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from model_inference_service import run_model_inference
from preprocessing_service import preprocess_text
from topic_modeling_service import train_lda_model, print_topics
from fastapi.staticfiles import StaticFiles  # Import StaticFiles here
import pandas as pd

#cd backend
#py -m uvicorn main:app --reload

app = FastAPI()

# Mounting the static directory
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Load the dataset and preprocess texts
@app.on_event("startup")
async def startup_event():
    df = pd.read_csv('C:/Users/DNS/Downloads/review_analysis/pythonProject/backend/translated_dataset1.csv')  # Replace with your dataset path
    preprocessed_texts = df['clean_translated_content'].apply(preprocess_text).tolist()
    train_lda_model(preprocessed_texts)

    # Print the topics
    topics = print_topics(n_top_words=10)
    for topic in topics:
        print(topic)


class Review(BaseModel):
    review_text: str = Field(..., example="I love this app, it has excellent features!")

@app.post("/analyze_review/", response_description="Returns the dominant topic from the review.")
async def analyze_review(review: Review):
    try:
        topic = run_model_inference(review.review_text)
        return {"dominant_topic": topic}
    except Exception as e:
        # Log the full exception
        print(f"Error: {e}")
        # Return a more descriptive error message
        raise HTTPException(status_code=500, detail=str(e))
