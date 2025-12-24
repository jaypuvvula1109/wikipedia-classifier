# Wikipedia Article Classifier

An end-to-end ML pipeline for classifying text queries and returning relevant Wikipedia article links. Built with **Ray** (Data, Train, Serve) and deployed on **Kubernetes** with **KubeRay**.

## 🎯 What It Does
```
User Query: "How do neural networks learn from data?"
     │
     ▼
┌─────────────────────────────────────┐
│         ML Classification           │
│    (TF-IDF + SGDClassifier)         │
└─────────────────────────────────────┘
     │
     ▼
Response:
  Category: Technology (87% confidence)
  Related Articles:
    • Artificial Intelligence
    • Neural Network
    • Machine Learning
```

## 🏗️ Architecture
```
┌──────────────────────────────────────────────────────────────────────┐
│                         Kubernetes (KubeRay)                         │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                        Ray Cluster                             │  │
│  │                                                                │  │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │  │
│  │   │  Head Node  │    │   Worker    │    │   Worker    │       │  │
│  │   │             │    │    Node     │    │    Node     │       │  │
│  │   │ • Ray Serve │    │             │    │             │       │  │
│  │   │ • Dashboard │    │ • Tasks     │    │ • Tasks     │       │  │
│  │   └─────────────┘    └─────────────┘    └─────────────┘       │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Data Processing | Ray Data |
| Model Training | Ray Train + scikit-learn |
| Model Serving | Ray Serve + FastAPI |
| Containerization | Docker |
| Orchestration | Kubernetes + KubeRay |
| Local K8s | Kind |

## 📁 Project Structure
```
wikipedia-classifier/
├── Dockerfile                 # Container image
├── requirements.txt           # Python dependencies
├── README.md
│
├── data/
│   └── articles.json          # Raw Wikipedia data
│
├── data_prep/
│   ├── download_data.py       # Download from Hugging Face
│   └── preprocess_ray_data.py # Process with Ray Data
│
├── training/
│   ├── train_classifier.py    # Train with Ray Train
│   └── improve_accuracy.py    # Hyperparameter tuning
│
├── models/
│   └── text_classifier.pkl    # Trained model
│
├── serving/
│   ├── serve.py               # Ray Serve API
│   └── test_api.py            # API test client
│
└── kuberay/
    ├── setup_kind.sh          # Create K8s cluster
    ├── install_kuberay.sh     # Install KubeRay operator
    ├── deploy_app.sh          # Build & deploy
    ├── cleanup.sh             # Delete everything
    └── ray-service.yaml       # RayService config
```

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.9+
python --version

# Docker
docker --version

# Kubernetes tools (for deployment)
kubectl version
kind --version
helm version
```

### Installation
```bash
# Clone repo
git clone https://github.com/jaypuvvula1109/wikipedia-classifier.git
cd wikipedia-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

### Step 1: Download Data
```bash
# From Hugging Face (recommended)
python data_prep/download_data.py

# Or use sample data
python data_prep/download_data.py --sample
```

### Step 2: Preprocess with Ray Data
```bash
python data_prep/preprocess_ray_data.py --source huggingface
```

### Step 3: Train Model
```bash
python training/train_classifier.py
```

### Step 4: Serve Locally
```bash
# Start server
python serving/serve.py

# Test (in another terminal)
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "How do plants convert sunlight to energy?", "top_k": 3}'
```

### Step 5: Deploy on Kubernetes
```bash
# Make scripts executable
chmod +x kuberay/*.sh

# Create Kind cluster
./kuberay/setup_kind.sh

# Install KubeRay operator
./kuberay/install_kuberay.sh

# Build and deploy
./kuberay/deploy_app.sh

# Access API
kubectl port-forward svc/wikipedia-classifier-serve-svc 8000:8000
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify text and get Wikipedia links |
| `/health` | GET | Health check |
| `/categories` | GET | List available categories |
| `/docs` | GET | Swagger UI documentation |

### Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "How do neural networks learn?", "top_k": 3}'
```

### Example Response
```json
{
  "query": "How do neural networks learn?",
  "predicted_category": "Technology",
  "confidence": 0.8734,
  "related_articles": [
    {
      "title": "Artificial intelligence",
      "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
      "relevance_score": 0.95
    },
    {
      "title": "Neural network",
      "url": "https://en.wikipedia.org/wiki/Neural_network",
      "relevance_score": 0.90
    },
    {
      "title": "Machine learning",
      "url": "https://en.wikipedia.org/wiki/Machine_learning",
      "relevance_score": 0.85
    }
  ]
}
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~88% |
| Categories | 5 (Science, Technology, History, Geography, Sports) |
| Model | TF-IDF + SGDClassifier |

### Improving Accuracy
```bash
python training/improve_accuracy.py
```

Methods to boost accuracy:
- Better TF-IDF parameters
- Try different classifiers (LinearSVC, LogisticRegression)
- Ensemble methods
- More training data

## 🧹 Cleanup
```bash
# Delete Kubernetes resources
./kuberay/cleanup.sh

# Or manually
kubectl delete rayservice wikipedia-classifier
kind delete cluster --name wikipedia-classifier
```

## 📚 Ray Concepts Used

| Concept | Purpose |
|---------|---------|
| `ray.data.read_json()` | Distributed data loading |
| `ds.map_batches()` | Parallel preprocessing |
| `ray.train` | Distributed model training |
| `@serve.deployment` | Deploy model as service |
| `@serve.ingress` | Connect FastAPI to Ray Serve |
| `RayService` | Kubernetes-native Ray deployment |

## 🔗 Resources

- [Ray Documentation](https://docs.ray.io/)
- [KubeRay Documentation](https://ray-project.github.io/kuberay/)
- [Ray Serve Guide](https://docs.ray.io/en/latest/serve/index.html)

## 📝 License

MIT License

## 🤝 Contributing

Pull requests welcome! Please open an issue first to discuss changes.
