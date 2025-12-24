

import ray
from ray import train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig

import json
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# =============================================================================
# STEP 1: Initialize Ray
# =============================================================================

def init_ray():
    """Initialize Ray cluster."""
    print("="*60)
    print("STEP 4.1: Initialize Ray")
    print("="*60)
    
    ray.init(ignore_reinit_error=True)
    
    print(f"✓ Ray initialized")
    print(f"  CPUs available: {ray.cluster_resources().get('CPU', 0)}")
    print()


# =============================================================================
# STEP 2: Load Data
# =============================================================================

def load_data(train_path="data/processed/train.json", test_path="data/processed/test.json"):
    """Load preprocessed train and test data."""
    print("="*60)
    print("STEP 4.2: Load Preprocessed Data")
    print("="*60)
    
    with open(train_path, "r") as f:
        train_data = json.load(f)
    
    with open(test_path, "r") as f:
        test_data = json.load(f)
    
    # Extract text and labels
    X_train = [item["text_clean"] for item in train_data]
    y_train = [item["category"] for item in train_data]
    
    X_test = [item["text_clean"] for item in test_data]
    y_test = [item["category"] for item in test_data]
    
    print(f"✓ Loaded data")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples:  {len(X_test)}")
    print(f"  Categories:    {sorted(set(y_train))}")
    print()
    
    return X_train, y_train, X_test, y_test


# =============================================================================
# STEP 3: Define Training Function
# =============================================================================

def train_text_classifier(X_train, y_train, X_test, y_test):
    """
    Train a text classification model.
    
    Pipeline:
    1. TfidfVectorizer - Convert text to TF-IDF features
    2. SGDClassifier   - Train classifier
    
    Returns trained model and metrics.
    """
    print("="*60)
    print("STEP 4.3: Train Text Classifier")
    print("="*60)
    
    # Create a pipeline: TF-IDF + Classifier
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,      # Limit vocabulary size
            ngram_range=(1, 2),     # Use unigrams and bigrams
            stop_words='english'    # Remove common words
        )),
        ('classifier', SGDClassifier(
            loss='log_loss',        # Logistic regression loss
            max_iter=1000,
            random_state=42
        ))
    ])
    
    print("Training model...")
    print("  Model: TF-IDF + SGDClassifier")
    print("  Features: up to 5000 (unigrams + bigrams)")
    
    # Train the model
    model.fit(X_train, y_train)
    
    print("✓ Training complete")
    print()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ Test Accuracy: {accuracy:.2%}")
    print()
    
    # Detailed classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, accuracy


# =============================================================================
# STEP 4: Save Model
# =============================================================================

def save_model(model, output_dir="models"):
    """Save the trained model to disk."""
    print("="*60)
    print("STEP 4.4: Save Model")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "text_classifier.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"✓ Model saved to {model_path}")
    print()
    
    return model_path


# =============================================================================
# STEP 5: Test Model
# =============================================================================

def test_model(model):
    """Test the model with sample predictions."""
    print("="*60)
    print("STEP 4.5: Test Model Predictions")
    print("="*60)
    
    test_texts = [
        "machine learning algorithms process data and make predictions",
        "the roman empire conquered many territories in europe",
        "photosynthesis converts sunlight into chemical energy in plants",
        "the championship game was won in overtime",
        "mount everest is the tallest mountain in the world"
    ]
    
    print("Sample predictions:\n")
    for text in test_texts:
        prediction = model.predict([text])[0]
        # Get prediction probabilities if available
        try:
            proba = model.predict_proba([text])[0]
            confidence = max(proba)
            print(f"  Text: \"{text[:50]}...\"")
            print(f"  → Predicted: {prediction} (confidence: {confidence:.2%})")
        except:
            print(f"  Text: \"{text[:50]}...\"")
            print(f"  → Predicted: {prediction}")
        print()


# =============================================================================
# STEP 6: Ray Train Version (Distributed Training)
# =============================================================================

def train_with_ray_train(X_train, y_train, X_test, y_test, output_dir="models"):
    """
    Train using Ray Train for distributed training.
    
    This is the Ray-native way to train models. Benefits:
    - Scales to multiple workers
    - Built-in checkpointing
    - Fault tolerance
    - Integrates with Ray ecosystem
    """
    print("="*60)
    print("STEP 4.6: Train with Ray Train")
    print("="*60)
    
    # Define the training function that runs on each worker
    def training_function(config):
        """
        This function runs on each Ray worker.
        For sklearn, we typically use 1 worker since sklearn 
        handles parallelism internally.
        """
        # Create model pipeline
        model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=config.get("max_features", 5000),
                ngram_range=(1, 2),
                stop_words='english'
            )),
            ('classifier', SGDClassifier(
                loss='log_loss',
                max_iter=config.get("max_iter", 1000),
                random_state=42
            ))
        ])
        
        # Train
        model.fit(config["X_train"], config["y_train"])
        
        # Evaluate
        y_pred = model.predict(config["X_test"])
        accuracy = accuracy_score(config["y_test"], y_pred)
        
        # Save model to checkpoint
        os.makedirs("model_checkpoint", exist_ok=True)
        model_path = "model_checkpoint/model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Report metrics and checkpoint to Ray Train
        train.report(
            {"accuracy": accuracy},
            checkpoint=train.Checkpoint.from_directory("model_checkpoint")
        )
    
    # Configure training
    trainer = ray.train.trainer.BasicTrainer(
        train_loop_per_worker=training_function,
        train_loop_config={
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "max_features": 5000,
            "max_iter": 1000,
        },
        scaling_config=ScalingConfig(
            num_workers=1,          # sklearn works best with 1 worker
            use_gpu=False,
        ),
        run_config=RunConfig(
            name="wikipedia_classifier",
            storage_path=os.path.abspath(output_dir),
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,      # Keep only best checkpoint
            ),
        ),
    )
    
    print("Starting Ray Train...")
    print("  Workers: 1")
    print("  GPU: False")
    print()
    
    # Run training
    result = trainer.fit()
    
    print(f"\n✓ Ray Train complete")
    print(f"  Accuracy: {result.metrics['accuracy']:.2%}")
    print(f"  Checkpoint: {result.checkpoint}")
    
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*60)
    print("STEP 4: Train Model with Ray Train")
    print("="*60 + "\n")
    
    # Initialize Ray
    init_ray()
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Simple training (without Ray Train)
    # Good for understanding the basics
    model, accuracy = train_text_classifier(X_train, y_train, X_test, y_test)
    
    # Save model
    model_path = save_model(model)
    
    # Test predictions
    test_model(model)
    
    # Optional: Train with Ray Train
    # Uncomment below to use Ray Train's distributed training
    # result = train_with_ray_train(X_train, y_train, X_test, y_test)
    
    # Summary
    print("="*60)
    print("STEP 4 COMPLETE: Model Training")
    print("="*60)
    print(f"""
What we learned:
  • TfidfVectorizer - Convert text to numerical features
  • SGDClassifier   - Fast, scalable classifier
  • Pipeline        - Chain preprocessing + model
  • Ray Train       - Distributed training framework
  
Model saved to:
  • {model_path}
  
Test Accuracy: {accuracy:.2%}

Next step:
  • Step 5: Serve model with Ray Serve
""")
    
    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()