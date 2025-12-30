from pymongo import MongoClient, ASCENDING, TEXT
from datasets import load_dataset
import argparse

# DBpedia category mapping
DBPEDIA_CATEGORIES = {
    0: "Company",
    1: "EducationalInstitution", 
    2: "Artist",
    3: "Athlete",
    4: "OfficeHolder",
    5: "MeanOfTransportation",
    6: "Building",
    7: "NaturalPlace",
    8: "Village",
    9: "Animal",
    10: "Plant",
    11: "Album",
    12: "Film",
    13: "WrittenWork"
}

# Simplified category mapping (group similar categories)
SIMPLIFIED_CATEGORIES = {
    0: "Business",           # Company
    1: "Education",          # EducationalInstitution
    2: "Arts",               # Artist
    3: "Sports",             # Athlete
    4: "Politics",           # OfficeHolder
    5: "Technology",         # MeanOfTransportation
    6: "Architecture",       # Building
    7: "Geography",          # NaturalPlace
    8: "Geography",          # Village
    9: "Science",            # Animal
    10: "Science",           # Plant
    11: "Arts",              # Album
    12: "Arts",              # Film
    13: "Literature"         # WrittenWork
}


def connect_mongodb(uri="mongodb://localhost:27017"):
    """Connect to MongoDB."""
    client = MongoClient(uri)
    db = client["wikipedia_classifier"]
    return db


def create_indexes(collection):
    """Create indexes for fast querying."""
    print("Creating indexes...")
    
    # Index on category for filtering
    collection.create_index([("category", ASCENDING)])
    
    # Index on label_id
    collection.create_index([("label_id", ASCENDING)])
    
    # Text index for full-text search
    collection.create_index([("title", TEXT), ("content", TEXT)])
    
    print("✓ Indexes created")


def load_data_to_mongodb(num_samples_per_class=500, uri="mongodb://localhost:27017"):
    """
    Load Hugging Face DBpedia data into MongoDB.
    
    Args:
        num_samples_per_class: Number of articles per category to load
        uri: MongoDB connection string
    """
    print("="*60)
    print("Loading Hugging Face Data into MongoDB")
    print("="*60)
    
    # Connect to MongoDB
    db = connect_mongodb(uri)
    collection = db["articles"]
    
    # Clear existing data
    print("\nClearing existing data...")
    collection.delete_many({})
    
    # Load from Hugging Face
    print("\nDownloading DBpedia dataset from Hugging Face...")
    print("(This may take a few minutes on first run)")
    dataset = load_dataset("dbpedia_14", split="train")
    print(f"✓ Dataset loaded: {len(dataset)} total articles")
    
    # Process and insert by category
    print(f"\nInserting {num_samples_per_class} articles per category...")
    
    total_inserted = 0
    
    for label_id in range(14):
        category = DBPEDIA_CATEGORIES[label_id]
        simplified = SIMPLIFIED_CATEGORIES[label_id]
        
        # Filter by label
        category_data = dataset.filter(lambda x: x['label'] == label_id)
        
        # Take samples
        num_to_take = min(num_samples_per_class, len(category_data))
        samples = category_data.select(range(num_to_take))
        
        # Prepare documents
        documents = []
        for sample in samples:
            doc = {
                "title": sample["title"],
                "content": sample["content"],
                "category": simplified,
                "original_category": category,
                "label_id": label_id,
                "url": f"https://en.wikipedia.org/wiki/{sample['title'].replace(' ', '_')}"
            }
            documents.append(doc)
        
        # Insert batch
        if documents:
            collection.insert_many(documents)
            total_inserted += len(documents)
        
        print(f"  ✓ {category}: {len(documents)} articles → {simplified}")
    
    # Create indexes
    print()
    create_indexes(collection)
    
    # Summary
    print("\n" + "="*60)
    print("✓ Data Loading Complete!")
    print("="*60)
    print(f"  Total articles: {total_inserted}")
    print(f"  Categories: {len(set(SIMPLIFIED_CATEGORIES.values()))}")
    print(f"  Database: wikipedia_classifier")
    print(f"  Collection: articles")
    
    # Show category counts
    print("\nArticles per category:")
    pipeline = [
        {"$group": {"_id": "$category", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    for doc in collection.aggregate(pipeline):
        print(f"  {doc['_id']}: {doc['count']}")
    
    return total_inserted


def test_mongodb(uri="mongodb://localhost:27017"):
    """Test MongoDB connection and data."""
    print("\n" + "="*60)
    print("Testing MongoDB Connection")
    print("="*60)
    
    db = connect_mongodb(uri)
    collection = db["articles"]
    
    # Count
    count = collection.count_documents({})
    print(f"✓ Total documents: {count}")
    
    # Sample query
    print("\nSample article:")
    sample = collection.find_one({"category": "Science"})
    if sample:
        print(f"  Title: {sample['title']}")
        print(f"  Category: {sample['category']}")
        print(f"  URL: {sample['url']}")
        print(f"  Content: {sample['content'][:100]}...")
    
    # Test text search
    print("\nText search for 'neural network':")
    results = collection.find(
        {"$text": {"$search": "neural network"}},
        {"score": {"$meta": "textScore"}}
    ).sort([("score", {"$meta": "textScore"})]).limit(3)
    
    for doc in results:
        print(f"  • {doc['title']} ({doc['category']})")


def main():
    parser = argparse.ArgumentParser(description="Load data to MongoDB")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of samples per category (default: 500)")
    parser.add_argument("--uri", default="mongodb://localhost:27017",
                        help="MongoDB connection URI")
    parser.add_argument("--test-only", action="store_true",
                        help="Only test connection, don't load data")
    args = parser.parse_args()
    
    if args.test_only:
        test_mongodb(args.uri)
    else:
        load_data_to_mongodb(args.samples, args.uri)
        test_mongodb(args.uri)


if __name__ == "__main__":
    main()