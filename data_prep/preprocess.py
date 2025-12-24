import ray
import re
import os
import argparse

# =============================================================================
# STEP 1: Initialize Ray
# =============================================================================

def init_ray():
    """Initialize Ray cluster."""
    print("="*60)
    print("STEP 3.1: Initialize Ray")
    print("="*60)
    
    # Initialize Ray (ignore if already running)
    ray.init(ignore_reinit_error=True)
    
    print(f"✓ Ray initialized")
    print(f"  CPUs available: {ray.cluster_resources().get('CPU', 0)}")
    print()


# =============================================================================
# STEP 2: Load Data with Ray Data
# =============================================================================

# DBpedia category names (for Hugging Face dataset)
DBPEDIA_CATEGORIES = [
    "Company", "EducationalInstitution", "Artist", "Athlete",
    "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
    "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"
]


def load_from_huggingface(num_samples_per_class=50):
    """
    Load data directly from Hugging Face into Ray Dataset.
    This is the recommended approach for large datasets!
    """
    from datasets import load_dataset
    
    print("Loading DBpedia dataset from Hugging Face...")
    
    # Load from Hugging Face
    hf_dataset = load_dataset("dbpedia_14", split="train")
    
    print(f"✓ Hugging Face dataset loaded: {len(hf_dataset)} total rows")
    
    # Sample from each category
    sampled_data = []
    for label_id, category in enumerate(DBPEDIA_CATEGORIES):
        # Filter by category
        category_data = hf_dataset.filter(lambda x: x['label'] == label_id)
        # Take samples
        samples = category_data.select(range(min(num_samples_per_class, len(category_data))))
        
        for sample in samples:
            sampled_data.append({
                "title": sample['title'],
                "text": sample['content'],
                "category": category,
                "url": f"https://en.wikipedia.org/wiki/{sample['title'].replace(' ', '_')}"
            })
        print(f"  ✓ {category}: {len(samples)} articles")
    
    # Convert to Ray Dataset
    ds = ray.data.from_items(sampled_data)
    
    print(f"\n✓ Created Ray Dataset with {ds.count()} rows")
    
    return ds


def load_from_json(data_path="data/articles.json"):
    """Load from local JSON file into Ray Dataset."""
    
    print(f"Loading from local JSON file: {data_path}")
    
    # Read JSON file into Ray Dataset
    ds = ray.data.read_json(data_path)
    
    print(f"✓ Loaded dataset from {data_path}")
    print(f"  Number of rows: {ds.count()}")
    
    return ds


def load_data(source="huggingface", json_path="data/articles.json", num_samples=50):
    """
    Load data into a Ray Dataset.
    
    Args:
        source: "huggingface" or "json"
        json_path: Path to JSON file (if source="json")
        num_samples: Samples per category (if source="huggingface")
    """
    print("="*60)
    print("STEP 3.2: Load Data with Ray Data")
    print("="*60)
    
    if source == "huggingface":
        try:
            ds = load_from_huggingface(num_samples_per_class=num_samples)
        except Exception as e:
            print(f"⚠ Could not load from Hugging Face: {e}")
            print("Falling back to local JSON file...")
            ds = load_from_json(json_path)
    else:
        ds = load_from_json(json_path)
    
    print(f"\nSchema: {ds.schema()}")
    
    # Show a sample
    print("\nSample row:")
    sample = ds.take(1)[0]
    for key, value in sample.items():
        if key == "text":
            print(f"  {key}: {value[:80]}...")
        else:
            print(f"  {key}: {value}")
    print()
    
    return ds


# =============================================================================
# STEP 3: Preprocess Data
# =============================================================================

def clean_text(text):
    """Clean a single text string."""
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters (keep letters, numbers, spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def preprocess_row(row):
    """
    Preprocess a single row. Used with ds.map()
    
    This function is called once per row.
    Good for simple transformations.
    """
    row["text_clean"] = clean_text(row["text"])
    row["text_length"] = len(row["text_clean"].split())
    return row


def preprocess_batch(batch):
    """
    Preprocess a batch of rows. Used with ds.map_batches()
    
    This function receives a dict of arrays (or pandas DataFrame).
    More efficient than map() for larger datasets.
    """
    # batch["text"] is a list of text strings
    batch["text_clean"] = [clean_text(t) for t in batch["text"]]
    batch["text_length"] = [len(t.split()) for t in batch["text_clean"]]
    return batch


def preprocess_data(ds, use_batches=True):
    """Apply preprocessing to the dataset."""
    print("="*60)
    print("STEP 3.3: Preprocess Data")
    print("="*60)
    
    if use_batches:
        # map_batches is more efficient for large datasets
        print("Using map_batches() for batch processing...")
        ds_processed = ds.map_batches(preprocess_batch)
    else:
        # map processes one row at a time
        print("Using map() for row-by-row processing...")
        ds_processed = ds.map(preprocess_row)
    
    print("✓ Preprocessing complete")
    print()
    
    # Show sample of processed data
    print("Sample processed row:")
    sample = ds_processed.take(1)[0]
    print(f"  Original:  {sample['text'][:60]}...")
    print(f"  Cleaned:   {sample['text_clean'][:60]}...")
    print(f"  Word count: {sample['text_length']}")
    print()
    
    return ds_processed


# =============================================================================
# STEP 4: Explore Data
# =============================================================================

def explore_data(ds):
    """Explore the dataset using Ray Data operations."""
    print("="*60)
    print("STEP 3.4: Explore Data")
    print("="*60)
    
    # Count articles per category
    print("Articles per category:")
    
    # Group by category and count
    # Note: We'll do this with a simple aggregation
    categories = {}
    for row in ds.iter_rows():
        cat = row["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    print()
    
    # Get basic stats on text length
    print("Text length statistics:")
    lengths = [row["text_length"] for row in ds.iter_rows()]
    print(f"  Min words: {min(lengths)}")
    print(f"  Max words: {max(lengths)}")
    print(f"  Avg words: {sum(lengths) / len(lengths):.1f}")
    print()


# =============================================================================
# STEP 5: Split Data
# =============================================================================

def split_data(ds, test_size=0.2):
    """Split data into train and test sets."""
    print("="*60)
    print("STEP 3.5: Split Data (Train/Test)")
    print("="*60)
    
    # Shuffle the data first
    ds_shuffled = ds.random_shuffle()
    
    # Materialize all data to split it
    all_data = list(ds_shuffled.iter_rows())
    total = len(all_data)
    train_count = int(total * (1 - test_size))
    
    # Split the data
    train_data = all_data[:train_count]
    test_data = all_data[train_count:]
    
    # Convert back to Ray Datasets
    train_ds = ray.data.from_items(train_data)
    test_ds = ray.data.from_items(test_data)
    
    print(f"✓ Data split complete")
    print(f"  Train set: {len(train_data)} articles")
    print(f"  Test set:  {len(test_data)} articles")
    print()
    
    return train_ds, test_ds


# =============================================================================
# STEP 6: Save Processed Data
# =============================================================================

def save_data(train_ds, test_ds, output_dir="data/processed"):
    """Save processed datasets to disk."""
    print("="*60)
    print("STEP 3.6: Save Processed Data")
    print("="*60)
    
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to lists and save as JSON
    train_data = list(train_ds.iter_rows())
    test_data = list(test_ds.iter_rows())
    
    train_path = os.path.join(output_dir, "train.json")
    test_path = os.path.join(output_dir, "test.json")
    
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✓ Saved train data to {train_path}")
    print(f"✓ Saved test data to {test_path}")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process data with Ray Data")
    parser.add_argument("--source", choices=["huggingface", "json"], default="huggingface",
                        help="Data source: 'huggingface' (download) or 'json' (local file)")
    parser.add_argument("--json-path", default="data/articles.json",
                        help="Path to JSON file (if source=json)")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Samples per category (if source=huggingface)")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("STEP 3: Process Data with Ray Data")
    print("="*60 + "\n")
    
    # Initialize Ray
    init_ray()
    
    # Load data
    ds = load_data(
        source=args.source, 
        json_path=args.json_path,
        num_samples=args.num_samples
    )
    
    # Preprocess
    ds_processed = preprocess_data(ds, use_batches=True)
    
    # Explore
    explore_data(ds_processed)
    
    # Split
    train_ds, test_ds = split_data(ds_processed, test_size=0.2)
    
    # Save
    save_data(train_ds, test_ds)
    
    # Summary
    print("="*60)
    print("STEP 3 COMPLETE: Data Processing with Ray Data")
    print("="*60)
    print("""
What we learned:
  • ray.data.from_huggingface()  - Load directly from Hugging Face
  • ray.data.read_json()         - Load from JSON file
  • ds.map_batches()             - Process batches (efficient)
  • ds.random_shuffle()          - Shuffle data
  
Output files:
  • data/processed/train.json
  • data/processed/test.json
  
Usage:
  python preprocess_ray_data.py --source huggingface   # From Hugging Face
  python preprocess_ray_data.py --source json          # From local JSON
""")
    
    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()