

from datasets import load_dataset
import pickle
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


# Category mapping
CATEGORIES = {
    0: "Business",       # Company
    1: "Education",      # EducationalInstitution
    2: "Arts",           # Artist
    3: "Sports",         # Athlete
    4: "Politics",       # OfficeHolder
    5: "Technology",     # MeanOfTransportation
    6: "Architecture",   # Building
    7: "Geography",      # NaturalPlace
    8: "Geography",      # Village
    9: "Science",        # Animal
    10: "Science",       # Plant
    11: "Arts",          # Album
    12: "Arts",          # Film
    13: "Literature"     # WrittenWork
}


def preprocess_text(text):
    """Clean text."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()


def load_from_huggingface(samples_per_category=500):
    """Load data directly from Hugging Face."""
    print("="*60)
    print("Loading Data from Hugging Face")
    print("="*60)
    
    print("\nDownloading DBpedia dataset...")
    print("(This may take a few minutes on first run)\n")
    
    dataset = load_dataset("dbpedia_14", split="train")
    print(f"✓ Total articles available: {len(dataset)}")
    
    texts = []
    labels = []
    
    print(f"\nSampling {samples_per_category} articles per category...")
    
    for label_id in range(14):
        category = CATEGORIES[label_id]
        
        # Filter by label
        category_data = dataset.filter(lambda x: x['label'] == label_id)
        
        # Take samples
        num_samples = min(samples_per_category, len(category_data))
        samples = category_data.select(range(num_samples))
        
        for sample in samples:
            texts.append(preprocess_text(sample['content']))
            labels.append(category)
        
        print(f"  ✓ Label {label_id}: {num_samples} articles → {category}")
    
    print(f"\n✓ Total loaded: {len(texts)} articles")
    
    return texts, labels


def train_model(X_train, y_train, X_test, y_test):
    """Train the model."""
    print("\n" + "="*60)
    print("Training Model")
    print("="*60)
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ('classifier', LinearSVC(
            C=1.0,
            max_iter=3000,
            random_state=42
        ))
    ])
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print("Training...")
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✓ Test Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, accuracy


def save_model(model, path="models/text_classifier.pkl"):
    """Save model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n✓ Model saved to {path}")


def test_model(model):
    """Test with sample predictions."""
    print("\n" + "="*60)
    print("Sample Predictions")
    print("="*60)
    
    tests = [
        "The French Revolution changed European politics",
        "Neural networks can learn patterns from data",
        "The World Cup is the biggest football tournament",
        "Mount Everest is the tallest mountain",
        "Shakespeare wrote Romeo and Juliet",
    ]
    
    for text in tests:
        pred = model.predict([preprocess_text(text)])[0]
        print(f"\n  \"{text[:50]}...\"")
        print(f"  → {pred}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500,
                        help="Samples per category (default: 500)")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("TRAIN FROM HUGGING FACE (No MongoDB)")
    print("="*60 + "\n")
    
    # Load directly from Hugging Face
    texts, labels = load_from_huggingface(args.samples)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    # Train
    model, accuracy = train_model(X_train, y_train, X_test, y_test)
    
    # Save
    save_model(model)
    
    # Test
    test_model(model)
    
    print("\n" + "="*60)
    print(f"DONE! Accuracy: {accuracy:.2%}")
    print("="*60)


if __name__ == "__main__":
    main()