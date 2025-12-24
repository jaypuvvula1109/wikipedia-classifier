
import json
import os
import argparse

# =============================================================================
# OPTION 1: HUGGING FACE DATASET
# =============================================================================

def download_from_huggingface(output_dir="data", num_samples_per_class=50):
    """
    Download DBpedia dataset from Hugging Face.
    DBpedia is structured data extracted from Wikipedia.
    """
    from datasets import load_dataset
    
    print("Downloading DBpedia dataset from Hugging Face...")
    print("(This may take a minute on first run)\n")
    
    dataset = load_dataset("dbpedia_14")
    
    print(f"✓ Dataset loaded!")
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Test samples: {len(dataset['test'])}")
    
    CATEGORIES = [
        "Company", "EducationalInstitution", "Artist", "Athlete",
        "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
        "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    all_articles = []
    
    print(f"\nSampling {num_samples_per_class} articles per category...")
    
    for label_id, category in enumerate(CATEGORIES):
        category_data = dataset['train'].filter(lambda x: x['label'] == label_id)
        samples = category_data.select(range(min(num_samples_per_class, len(category_data))))
        
        for sample in samples:
            all_articles.append({
                "title": sample['title'],
                "text": sample['content'],
                "category": category,
                "url": f"https://en.wikipedia.org/wiki/{sample['title'].replace(' ', '_')}"
            })
        
        print(f"  ✓ {category}: {len(samples)} articles")
    
    return all_articles, CATEGORIES


# =============================================================================
# OPTION 2: SAMPLE DATA (BUILT-IN)
# =============================================================================

SAMPLE_DATA = {
    "Science": [
        {"title": "Photosynthesis", "text": "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy. This chemical energy is stored in carbohydrate molecules such as sugars. Photosynthesis is largely responsible for producing and maintaining the oxygen content of the Earth's atmosphere.", "url": "https://en.wikipedia.org/wiki/Photosynthesis"},
        {"title": "DNA", "text": "Deoxyribonucleic acid is a molecule composed of two polynucleotide chains that coil around each other to form a double helix. DNA carries genetic instructions for the development, functioning, growth and reproduction of all known organisms and many viruses.", "url": "https://en.wikipedia.org/wiki/DNA"},
        {"title": "Black hole", "text": "A black hole is a region of spacetime where gravity is so strong that nothing can escape from it. The theory of general relativity predicts that a sufficiently compact mass can deform spacetime to form a black hole.", "url": "https://en.wikipedia.org/wiki/Black_hole"},
        {"title": "Evolution", "text": "Evolution is change in the heritable characteristics of biological populations over successive generations. Natural selection is a key mechanism of evolution.", "url": "https://en.wikipedia.org/wiki/Evolution"},
        {"title": "Quantum mechanics", "text": "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles.", "url": "https://en.wikipedia.org/wiki/Quantum_mechanics"},
    ],
    "Technology": [
        {"title": "Artificial intelligence", "text": "Artificial intelligence is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research focuses on creating systems that can learn, reason, and solve problems.", "url": "https://en.wikipedia.org/wiki/Artificial_intelligence"},
        {"title": "Neural network", "text": "A neural network is a computing system inspired by biological neural networks in animal brains. Neural networks learn to perform tasks by considering examples.", "url": "https://en.wikipedia.org/wiki/Neural_network"},
        {"title": "Blockchain", "text": "A blockchain is a distributed ledger with growing lists of records called blocks that are securely linked together. Blockchains are used for cryptocurrencies like Bitcoin.", "url": "https://en.wikipedia.org/wiki/Blockchain"},
        {"title": "Cloud computing", "text": "Cloud computing is the delivery of computing services over the internet including servers, storage, databases, networking, software, and analytics.", "url": "https://en.wikipedia.org/wiki/Cloud_computing"},
        {"title": "Machine learning", "text": "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.", "url": "https://en.wikipedia.org/wiki/Machine_learning"},
    ],
    "History": [
        {"title": "World War II", "text": "World War II was a global conflict that lasted from 1939 to 1945. It involved the vast majority of the world's countries forming two opposing military alliances: the Allies and the Axis powers.", "url": "https://en.wikipedia.org/wiki/World_War_II"},
        {"title": "Ancient Rome", "text": "Ancient Rome refers to the civilization that grew from a small town on the Tiber River into an empire that encompassed most of continental Europe, Britain, western Asia, and North Africa.", "url": "https://en.wikipedia.org/wiki/Ancient_Rome"},
        {"title": "French Revolution", "text": "The French Revolution was a period of radical political and societal change in France that began in 1789. It marked the end of absolute monarchy and established a republic.", "url": "https://en.wikipedia.org/wiki/French_Revolution"},
        {"title": "Industrial Revolution", "text": "The Industrial Revolution was the transition from hand production methods to machine manufacturing. It began in Britain in the 18th century.", "url": "https://en.wikipedia.org/wiki/Industrial_Revolution"},
        {"title": "Renaissance", "text": "The Renaissance was a cultural movement that began in Italy in the 14th century. It marked the transition from the medieval period to modernity.", "url": "https://en.wikipedia.org/wiki/Renaissance"},
    ],
    "Geography": [
        {"title": "Amazon rainforest", "text": "The Amazon rainforest is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. It represents over half of the planet's remaining rainforests.", "url": "https://en.wikipedia.org/wiki/Amazon_rainforest"},
        {"title": "Mount Everest", "text": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas.", "url": "https://en.wikipedia.org/wiki/Mount_Everest"},
        {"title": "Pacific Ocean", "text": "The Pacific Ocean is the largest and deepest ocean on Earth. It covers about 46 percent of Earth's water surface.", "url": "https://en.wikipedia.org/wiki/Pacific_Ocean"},
        {"title": "Sahara Desert", "text": "The Sahara is a desert on the African continent. It is the largest hot desert in the world and the third largest desert overall.", "url": "https://en.wikipedia.org/wiki/Sahara"},
        {"title": "Grand Canyon", "text": "The Grand Canyon is a steep-sided canyon carved by the Colorado River in Arizona, United States. It is considered one of the Seven Natural Wonders of the World.", "url": "https://en.wikipedia.org/wiki/Grand_Canyon"},
    ],
    "Sports": [
        {"title": "Olympic Games", "text": "The Olympic Games are international athletic competitions featuring summer and winter sports. The modern Olympics began in Athens in 1896.", "url": "https://en.wikipedia.org/wiki/Olympic_Games"},
        {"title": "Football", "text": "Football, also called soccer, is a team sport played between two teams of eleven players. It is the world's most popular sport.", "url": "https://en.wikipedia.org/wiki/Association_football"},
        {"title": "Basketball", "text": "Basketball is a team sport in which two teams of five players try to score points by throwing a ball through a hoop. The NBA is the premier professional basketball league.", "url": "https://en.wikipedia.org/wiki/Basketball"},
        {"title": "Tennis", "text": "Tennis is a racket sport played individually against a single opponent or between two teams of two players each.", "url": "https://en.wikipedia.org/wiki/Tennis"},
        {"title": "Cricket", "text": "Cricket is a bat-and-ball game played between two teams of eleven players on a circular field.", "url": "https://en.wikipedia.org/wiki/Cricket"},
    ]
}


def use_sample_data(output_dir="data"):
    """Use built-in sample data (works offline)."""
    
    print("Using built-in sample data...\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_articles = []
    categories = list(SAMPLE_DATA.keys())
    
    for category, articles in SAMPLE_DATA.items():
        for article in articles:
            all_articles.append({
                "title": article["title"],
                "text": article["text"],
                "category": category,
                "url": article["url"]
            })
        print(f"  ✓ {category}: {len(articles)} articles")
    
    return all_articles, categories


# =============================================================================
# MAIN
# =============================================================================

def save_data(articles, categories, output_dir="data"):
    """Save articles to JSON file."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    articles_path = os.path.join(output_dir, "articles.json")
    with open(articles_path, "w") as f:
        json.dump(articles, f, indent=2)
    
    print(f"\n✓ Saved {len(articles)} articles to {articles_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Total articles: {len(articles)}")
    print(f"Categories: {len(categories)}")
    print(f"\nCategories:")
    for cat in categories:
        count = len([a for a in articles if a['category'] == cat])
        print(f"  - {cat}: {count} articles")
    
    print("\n" + "="*60)
    print("Sample article:")
    print("="*60)
    sample = articles[0]
    print(f"Title:    {sample['title']}")
    print(f"Category: {sample['category']}")
    print(f"Text:     {sample['text'][:150]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Wikipedia data")
    parser.add_argument("--sample", action="store_true", 
                        help="Use sample data instead of downloading")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of samples per category (for Hugging Face)")
    args = parser.parse_args()
    
    print("="*60)
    print("STEP 2: Download Wikipedia Data")
    print("="*60 + "\n")
    
    if args.sample:
        # Use sample data
        articles, categories = use_sample_data()
    else:
        # Try Hugging Face first, fall back to sample data
        try:
            articles, categories = download_from_huggingface(
                num_samples_per_class=args.num_samples
            )
        except Exception as e:
            print(f"⚠ Could not download from Hugging Face: {e}")
            print("Falling back to sample data...\n")
            articles, categories = use_sample_data()
    
    save_data(articles, categories)
