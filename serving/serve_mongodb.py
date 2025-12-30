"""
Serve Model with Ray Serve + MongoDB
=====================================

This script creates a REST API that:
1. Classifies text using the trained model
2. Queries MongoDB for relevant Wikipedia articles
3. Returns articles sorted by relevance

API Endpoints:
  POST /predict         - Classify text and get Wikipedia links
  GET  /health          - Health check
  GET  /categories      - List available categories
  GET  /search          - Search articles by text
"""

import ray
from ray import serve
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from pymongo import MongoClient
import pickle
import re
import numpy as np
from typing import List, Optional


# =============================================================================
# Request/Response Models
# =============================================================================

class PredictRequest(BaseModel):
    text: str
    top_k: Optional[int] = 5

    class Config:
        json_schema_extra = {
            "example": {
                "text": "How do neural networks learn from data?",
                "top_k": 5
            }
        }


class ArticleLink(BaseModel):
    title: str
    url: str
    category: str
    relevance_score: float
    snippet: str


class PredictResponse(BaseModel):
    query: str
    predicted_category: str
    confidence: float
    related_articles: List[ArticleLink]


class SearchRequest(BaseModel):
    query: str
    category: Optional[str] = None
    limit: Optional[int] = 10


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Wikipedia Article Classifier API",
    description="Classify text and get relevant Wikipedia article links from MongoDB",
    version="2.0.0"
)


# =============================================================================
# Ray Serve Deployment
# =============================================================================

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1},
)
@serve.ingress(app)
class WikipediaClassifier:
    """
    Ray Serve deployment with MongoDB integration.
    """
    
    def __init__(
        self, 
        model_path: str = "models/text_classifier.pkl",
        mongodb_uri: str = "mongodb://localhost:27017"
    ):
        """Load model and connect to MongoDB."""
        print("="*50)
        print("Initializing Wikipedia Classifier")
        print("="*50)
        
        # Load ML model
        print(f"Loading model from {model_path}...")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.categories = list(self.model.classes_)
        print(f"✓ Model loaded! Categories: {self.categories}")
        
        # Connect to MongoDB
        print(f"\nConnecting to MongoDB at {mongodb_uri}...")
        self.mongo_client = MongoClient(mongodb_uri)
        self.db = self.mongo_client["wikipedia_classifier"]
        self.collection = self.db["articles"]
        
        # Test connection
        count = self.collection.count_documents({})
        print(f"✓ MongoDB connected! {count} articles available")
        
        print("\n" + "="*50)
        print("✓ Service ready!")
        print("="*50 + "\n")
    
    def _preprocess_text(self, text: str) -> str:
        """Clean input text."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()
    
    def _get_confidence(self, text: str) -> tuple:
        """Get prediction and confidence score."""
        # Predict
        prediction = self.model.predict([text])[0]
        
        # Get confidence
        try:
            # For LinearSVC, use decision_function
            decision = self.model.decision_function([text])[0]
            # Convert to probability using softmax
            exp_scores = np.exp(decision - np.max(decision))
            probs = exp_scores / exp_scores.sum()
            confidence = float(np.max(probs))
        except:
            try:
                # For models with predict_proba
                probs = self.model.predict_proba([text])[0]
                confidence = float(max(probs))
            except:
                confidence = 0.0
        
        return prediction, confidence
    
    def _search_articles(
        self, 
        query: str, 
        category: str = None, 
        limit: int = 5
    ) -> List[dict]:
        """
        Search MongoDB for relevant articles.
        
        Uses text search to find articles matching the query,
        optionally filtered by category.
        """
        # Build search query
        search_filter = {}
        
        # Add text search
        if query:
            search_filter["$text"] = {"$search": query}
        
        # Add category filter
        if category:
            search_filter["category"] = category
        
        # Execute search with text score
        try:
            if "$text" in search_filter:
                cursor = self.collection.find(
                    search_filter,
                    {"score": {"$meta": "textScore"}}
                ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            else:
                # No text search, just filter by category
                cursor = self.collection.find(search_filter).limit(limit)
            
            articles = []
            for doc in cursor:
                score = doc.get("score", 0.5)
                # Normalize score to 0-1 range
                normalized_score = min(score / 10.0, 1.0)
                
                articles.append({
                    "title": doc["title"],
                    "url": doc["url"],
                    "category": doc["category"],
                    "relevance_score": round(normalized_score, 3),
                    "snippet": doc["content"][:200] + "..."
                })
            
            return articles
            
        except Exception as e:
            print(f"Search error: {e}")
            # Fallback: just get articles from category
            cursor = self.collection.find(
                {"category": category} if category else {}
            ).limit(limit)
            
            articles = []
            for i, doc in enumerate(cursor):
                articles.append({
                    "title": doc["title"],
                    "url": doc["url"],
                    "category": doc["category"],
                    "relevance_score": round(0.9 - (i * 0.1), 3),
                    "snippet": doc["content"][:200] + "..."
                })
            
            return articles
    
    # -------------------------------------------------------------------------
    # API Endpoints
    # -------------------------------------------------------------------------
    
    @app.post("/predict", response_model=PredictResponse)
    async def predict(self, request: PredictRequest) -> PredictResponse:
        """
        Classify text and return relevant Wikipedia articles.
        """
        # Preprocess
        clean_text = self._preprocess_text(request.text)
        
        if not clean_text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Get prediction and confidence
        prediction, confidence = self._get_confidence(clean_text)
        
        # Search for related articles
        # Use original text for better search results
        articles = self._search_articles(
            query=request.text,
            category=prediction,
            limit=request.top_k
        )
        
        return PredictResponse(
            query=request.text,
            predicted_category=prediction,
            confidence=round(confidence, 4),
            related_articles=[ArticleLink(**a) for a in articles]
        )
    
    @app.get("/search")
    async def search(
        self, 
        q: str = Query(..., description="Search query"),
        category: Optional[str] = Query(None, description="Filter by category"),
        limit: int = Query(10, description="Max results")
    ):
        """
        Search articles by text query.
        """
        articles = self._search_articles(
            query=q,
            category=category,
            limit=limit
        )
        
        return {
            "query": q,
            "category_filter": category,
            "count": len(articles),
            "articles": articles
        }
    
    @app.get("/health")
    async def health(self):
        """Health check."""
        try:
            # Check MongoDB connection
            self.mongo_client.admin.command('ping')
            mongo_status = "connected"
            article_count = self.collection.count_documents({})
        except:
            mongo_status = "disconnected"
            article_count = 0
        
        return {
            "status": "healthy",
            "model_loaded": self.model is not None,
            "mongodb": mongo_status,
            "articles_count": article_count,
            "categories": self.categories
        }
    
    @app.get("/categories")
    async def get_categories(self):
        """List available categories with article counts."""
        pipeline = [
            {"$group": {"_id": "$category", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        category_counts = {}
        for doc in self.collection.aggregate(pipeline):
            category_counts[doc["_id"]] = doc["count"]
        
        return {
            "model_categories": self.categories,
            "database_categories": category_counts,
            "total_articles": sum(category_counts.values())
        }
    
    @app.get("/")
    async def root(self):
        """Root endpoint."""
        return {
            "name": "Wikipedia Article Classifier API",
            "version": "2.0.0",
            "features": [
                "Text classification with ML model",
                "MongoDB-backed article search",
                "Full-text search support"
            ],
            "endpoints": {
                "POST /predict": "Classify text and get Wikipedia links",
                "GET /search?q=query": "Search articles by text",
                "GET /health": "Health check",
                "GET /categories": "List categories",
                "GET /docs": "API documentation"
            }
        }


# =============================================================================
# Create Deployment
# =============================================================================

deployment = WikipediaClassifier.bind(
    model_path="models/text_classifier.pkl",
    mongodb_uri="mongodb://localhost:27017"
)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Wikipedia Classifier API (MongoDB Edition)")
    print("="*60 + "\n")
    
    ray.init(ignore_reinit_error=True)
    
    print("Starting Ray Serve...")
    print("  API: http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop.\n")
    
    serve.run(deployment, blocking=True)