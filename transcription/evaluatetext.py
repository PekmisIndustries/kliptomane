from transformers import pipeline
import re
from transformers.utils import logging
logging.set_verbosity_error()

# Load the French-specific sentiment analysis model
def load_model():
    """
    Load a French-specific sentiment analysis model.
    """
    #j-hartmann/emotion-english-distilroberta-base
    #nlptown/bert-large-multilingual-uncased-sentiment
    model_name = "j-hartmann/emotion-english-distilroberta-base"

    local_model = "sentiment/models--j-hartmann--emotion-english-distilroberta-base"
    print(f"Chargement du modele > {model_name}")
    print(f"Traitement\n--------------------------\\...............")
    return pipeline("text-classification", model=local_model)

# Function to clean and normalize text
def clean_text(text):
    return text.strip()

# Function to evaluate text using the French-specific model
def evaluate_text_locally(model, text):
    """
    Use the selected French-specific model to evaluate the text and assign a score.
    """
    try:
        result = model(text)[0]
        # Convert sentiment to a score
        if result["label"] == "POSITIVE":
            score = result["score"] * 5  # Scale positive scores to 0-5
        else:  # NEGATIVE or NEUTRAL
            score = (1 - result["score"]) * 5  # Scale negative scores inversely
    except Exception as e:
        print(f"Error evaluating text: {text}\n{e}")
        score = 0  # Assign 0 in case of error
    return score

# Function to evaluate intervals and assign scores
def evaluate_intervals_locally(transcription_file, output_file):
    # Load the French-specific model
    model = load_model()

    with open(transcription_file, "r") as f:
        lines = f.readlines()

    scored_intervals = []
    for line in lines:
        match = re.match(r"\[(\d+\.\d+) - (\d+\.\d+)\] (.+)", line)
        if match:
            start, end, text = match.groups()
            text = clean_text(text)

            # Skip empty or trivial intervals
            if text.__contains__("PEKMIGNORE"):
                # print("modele > ignore      0.00 |  *Silence")
                text = "---"
                score=0.00
            else:
                # Evaluate the interval using the selected model
                score = evaluate_text_locally(model, text)
                if(score>9.99):
                    score=9.99
                print("modele > traitement ", round(score, 2), "| ",text, " <|")
            scored_intervals.append((float(start), float(end), text, score))

    # Save scored intervals to a file
    with open(output_file, "w") as f:
        for interval in scored_intervals:
            start, end, text, score = interval
            f.write(f"{score:.2f} | [{start:.2f} - {end:.2f}] {text}\n")

    return scored_intervals

# Paths to transcription and output files
transcription_file = "transcription.txt"
output_file = "scored_intervals.txt"

# Evaluate intervals and save results
scored_intervals = evaluate_intervals_locally(transcription_file, output_file)
print(f"--------------------------/'''''''''''''''\nClassement émotionel terminé > {output_file}")
