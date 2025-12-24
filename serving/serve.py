import ray
from ray import serve
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import json
import os
from typing import List, Optional


# =============================================================================
# Request/Response Models (Pydantic)
# =============================================================================

class PredictRequest(BaseModel):
    """Input for prediction endpoint."""
    text: str
    top_k: Optional[int] = 3  # Number of article links to return
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "How do neural networks learn from data?",
                "top_k": 3
            }
        }


class ArticleLink(BaseModel):
    """A Wikipedia article link."""
    title: str
    url: str
    relevance_score: float


class PredictResponse(BaseModel):
    """Output from prediction endpoint."""
    query: str
    predicted_category: str
    confidence: float
    related_articles: List[ArticleLink]


# =============================================================================
# Wikipedia Article Database (for returning links)
# =============================================================================

# Sample articles per category (in production, use a real database)
WIKIPEDIA_ARTICLES = {
    "Science": [
        {"title": "Photosynthesis", "url": "https://en.wikipedia.org/wiki/Photosynthesis"},
        {"title": "DNA", "url": "https://en.wikipedia.org/wiki/DNA"},
        {"title": "Evolution", "url": "https://en.wikipedia.org/wiki/Evolution"},
        {"title": "Quantum mechanics", "url": "https://en.wikipedia.org/wiki/Quantum_mechanics"},
        {"title": "Black hole", "url": "https://en.wikipedia.org/wiki/Black_hole"},
    ],
    "Technology": [
        {"title": "Artificial intelligence", "url": "https://en.wikipedia.org/wiki/Artificial_intelligence"},
        {"title": "Neural network", "url": "https://en.wikipedia.org/wiki/Neural_network"},
        {"title": "Machine learning", "url": "https://en.wikipedia.org/wiki/Machine_learning"},
        {"title": "Blockchain", "url": "https://en.wikipedia.org/wiki/Blockchain"},
        {"title": "Cloud computing", "url": "https://en.wikipedia.org/wiki/Cloud_computing"},
    ],
    "History": [
        {"title": "World War II", "url": "https://en.wikipedia.org/wiki/World_War_II"},
        {"title": "Ancient Rome", "url": "https://en.wikipedia.org/wiki/Ancient_Rome"},
        {"title": "French Revolution", "url": "https://en.wikipedia.org/wiki/French_Revolution"},
        {"title": "Industrial Revolution", "url": "https://en.wikipedia.org/wiki/Industrial_Revolution"},
        {"title": "Renaissance", "url": "https://en.wikipedia.org/wiki/Renaissance"},
    ],
    "Geography": [
        {"title": "Amazon rainforest", "url": "https://en.wikipedia.org/wiki/Amazon_rainforest"},
        {"title": "Mount Everest", "url": "https://en.wikipedia.org/wiki/Mount_Everest"},
        {"title": "Pacific Ocean", "url": "https://en.wikipedia.org/wiki/Pacific_Ocean"},
        {"title": "Sahara Desert", "url": "https://en.wikipedia.org/wiki/Sahara"},
        {"title": "Grand Canyon", "url": "https://en.wikipedia.org/wiki/Grand_Canyon"},
    ],
    "Sports": [
        {"title": "Olympic Games", "url": "https://en.wikipedia.org/wiki/Olympic_Games"},
        {"title": "Football", "url": "https://en.wikipedia.org/wiki/Association_football"},
        {"title": "Basketball", "url": "https://en.wikipedia.org/wiki/Basketball"},
        {"title": "Tennis", "url": "https://en.wikipedia.org/wiki/Tennis"},
        {"title": "Cricket", "url": "https://en.wikipedia.org/wiki/Cricket"},
    ],
}


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Wikipedia Article Classifier API",
    description="Classify text and get relevant Wikipedia article links",
    version="1.0.0"
)


# =============================================================================
# Ray Serve Deployment
# =============================================================================

@serve.deployment(
    num_replicas=1,                    # Number of copies of the service
    ray_actor_options={"num_cpus": 1}, # Resources per replica
)
@serve.ingress(app)
class WikipediaClassifier:
    """
    Ray Serve deployment for Wikipedia classification.
    
    This class:
    1. Loads the trained model on startup
    2. Handles HTTP requests via FastAPI
    3. Returns predictions with Wikipedia links
    """
    
    def __init__(self, model_path: str = "models/text_classifier.pkl"):
        """Load model when deployment starts."""
        print(f"Loading model from {model_path}...")
        
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        
        # Get list of categories from model
        self.categories = list(self.model.classes_)
        
        print(f"✓ Model loaded! Categories: {self.categories}")
    
    def _preprocess_text(self, text: str) -> str:
        """Clean input text (same as training)."""
        import re
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()
    
    def _get_related_articles(self, category: str, top_k: int) -> List[dict]:
        """Get Wikipedia articles for a category."""
        articles = WIKIPEDIA_ARTICLES.get(category, [])
        
        # Add fake relevance scores (in production, compute real similarity)
        result = []
        for i, article in enumerate(articles[:top_k]):
            result.append({
                "title": article["title"],
                "url": article["url"],
                "relevance_score": round(0.95 - (i * 0.05), 2)  # Decreasing scores
            })
        
        return result
    
    # -------------------------------------------------------------------------
    # API Endpoints
    # -------------------------------------------------------------------------
    
    @app.post("/predict", response_model=PredictResponse)
    async def predict(self, request: PredictRequest) -> PredictResponse:
        """
        Classify text and return relevant Wikipedia articles.
        
        Example:
            POST /predict
            {"text": "How do plants convert sunlight to energy?"}
            
            Response:
            {
                "query": "How do plants convert sunlight to energy?",
                "predicted_category": "Science",
                "confidence": 0.89,
                "related_articles": [...]
            }
        """
        # Preprocess
        clean_text = self._preprocess_text(request.text)
        
        if not clean_text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Predict
        prediction = self.model.predict([clean_text])[0]
        
        # Get confidence (probability)
        try:
            probabilities = self.model.predict_proba([clean_text])[0]
            confidence = float(max(probabilities))
        except:
            confidence = 0.0  # Some models don't support predict_proba
        
        # Get related articles
        articles = self._get_related_articles(prediction, request.top_k)
        
        return PredictResponse(
            query=request.text,
            predicted_category=prediction,
            confidence=round(confidence, 4),
            related_articles=[ArticleLink(**a) for a in articles]
        )
    
    @app.get("/health")
    async def health(self):
        """Health check endpoint."""
        return {"status": "healthy", "model_loaded": self.model is not None}
    
    @app.get("/categories")
    async def get_categories(self):
        """List available categories."""
        return {
            "categories": self.categories,
            "count": len(self.categories)
        }
    
    @app.get("/")
    async def root(self):
        """Root endpoint with API info."""
        return {
            "name": "Wikipedia Article Classifier API",
            "version": "1.0.0",
            "endpoints": {
                "POST /predict": "Classify text and get Wikipedia links",
                "GET /health": "Health check",
                "GET /categories": "List categories",
                "GET /docs": "API documentation (Swagger UI)"
            }
        }


# =============================================================================
# Create Serve Application
# =============================================================================

# Bind deployment with model path
deployment = WikipediaClassifier.bind(model_path="models/text_classifier.pkl")


# =============================================================================
# Main: Run the Server
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("STEP 5: Serve Model with Ray Serve")
    print("="*60 + "\n")
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    print("Starting Ray Serve...")
    print("  API will be available at: http://localhost:8000")
    print("  Swagger docs at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server.\n")
    
    # Run the deployment
    serve.run(deployment, blocking=True)
