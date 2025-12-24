

import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("="*50)
    print("Testing: GET /health")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_categories():
    """Test categories endpoint."""
    print("="*50)
    print("Testing: GET /categories")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/categories")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_predict(text, top_k=3):
    """Test predict endpoint."""
    print("="*50)
    print(f"Testing: POST /predict")
    print(f"Input: \"{text}\"")
    print("="*50)
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"text": text, "top_k": top_k}
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nPredicted Category: {data['predicted_category']}")
        print(f"Confidence: {data['confidence']:.2%}")
        print(f"\nRelated Wikipedia Articles:")
        for article in data['related_articles']:
            print(f"  • {article['title']}")
            print(f"    {article['url']}")
            print(f"    Relevance: {article['relevance_score']}")
    else:
        print(f"Error: {response.text}")
    
    print()


def main():
    print("\n" + "="*50)
    print("WIKIPEDIA CLASSIFIER API TEST")
    print("="*50 + "\n")
    
    try:
        # Test health
        test_health()
        
        # Test categories
        test_categories()
        
        # Test predictions
        test_cases = [
            "How do neural networks learn from data?",
            "What caused World War II?",
            "How does photosynthesis work in plants?",
            "Who won the last Olympic Games?",
            "Where is Mount Everest located?",
        ]
        
        for text in test_cases:
            test_predict(text)
        
        print("="*50)
        print("ALL TESTS COMPLETE!")
        print("="*50)
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to API server.")
        print("\nMake sure the server is running:")
        print("  python 03_serving/serve_app.py")


if __name__ == "__main__":
    main()