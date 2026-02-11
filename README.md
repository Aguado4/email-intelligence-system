# Email Intelligence System

A production-ready microservices architecture for intelligent email classification using LangGraph, FastAPI, and LLM orchestration.

## 🏗️ Architecture
```
┌──────────────────┐
│   External       │
│   Client         │
└────────┬─────────┘
         │ HTTP
         ↓
┌─────────────────────────────────────────┐
│       API Gateway (FastAPI)             │
│  - Entry point for all requests         │
│  - Request validation                   │
│  - Response formatting                  │
└────────┬────────────────────────────────┘
         │ Internal HTTP
         ↓
┌─────────────────────────────────────────┐
│   Classifier Service (LangGraph)        │
│  - Email classification workflow        │
│  - Conditional routing based on conf    │
│  - Gemini 2.0 Flash integration         │
└─────────────────────────────────────────┘
```

## 🚀 Features

### Current Features ✅
- **Microservices Architecture**: Loosely coupled services with Docker orchestration
- **Async FastAPI**: High-performance async I/O for all HTTP endpoints
- **LangGraph Workflows**: Stateful, conditional email classification pipeline
- **Provider-Agnostic LLM**: Easy switching between Gemini, OpenAI, Anthropic
- **Confidence-Based Routing**: Automatic re-analysis for low-confidence classifications
- **Type Safety**: Pydantic models with validation throughout
- **Docker Compose**: One-command deployment
- **Health Monitoring**: Health check endpoints for all services
- **Structured Logging**: Comprehensive logging for debugging

### Planned Features 🚧
- **RAG Enhancement**: Vector database for few-shot learning on low-confidence cases
- **PostgreSQL Integration**: Persistent storage for classifications and analytics
- **Evaluation Service**: Automated testing with F1, precision, recall metrics
- **Grafana Dashboard**: Real-time monitoring and metrics visualization

## 🛠️ Tech Stack

- **Python 3.11+**
- **FastAPI**: Modern async web framework
- **LangGraph**: LLM workflow orchestration with state machines
- **LangChain**: LLM integrations and abstractions
- **Gemini 2.0 Flash**: Google's latest LLM (configurable)
- **Docker & Docker Compose**: Containerization and orchestration
- **Pydantic**: Data validation and settings management
- **httpx**: Async HTTP client for service communication

## 📦 Project Structure
```
email-intelligence-system/
├── services/
│   ├── api-gateway/              # FastAPI entry point
│   │   ├── app/
│   │   │   ├── main.py          # FastAPI app
│   │   │   ├── routes/          # API endpoints
│   │   │   └── clients/         # Service clients
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   └── classifier-service/       # LangGraph classifier
│       ├── app/
│       │   └── main.py          # FastAPI wrapper
│       ├── workflows/
│       │   └── email_classifier.py  # LangGraph workflow
│       ├── config/
│       │   └── settings.py      # Configuration
│       ├── Dockerfile
│       └── requirements.txt
│
├── shared/                       # Shared Pydantic models
│   └── models/
│       └── schemas.py
│
├── docker-compose.yml           # Service orchestration
├── .env.example                 # Environment template
└── README.md
```

## 🏃 Quick Start

### Prerequisites
- Docker Desktop installed
- Gemini API key (get from https://aistudio.google.com/app/apikey)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Aguado4/email-intelligence-system.git
cd email-intelligence-system
```

2. **Configure environment**
```bash
# Create .env file
cp .env.example .env

# Edit .env and add your API key
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=your-key-here
```

3. **Start all services**
```bash
docker-compose up
```

Services will be available at:
- **API Gateway**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (Swagger UI)

### Usage Examples

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Classify an Email
```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "email_id": "test_001",
    "subject": "URGENT: Account verification required",
    "body": "Click here to verify your account immediately...",
    "sender": "noreply@suspicious.com"
  }'
```

**Response:**
```json
{
  "email_id": "test_001",
  "classification": {
    "category": "spam",
    "confidence": 0.95,
    "reasoning": "Contains urgent language and suspicious sender",
    "keywords": ["urgent", "verify", "click"]
  },
  "processing_time_ms": 1234.56,
  "timestamp": "2025-12-16T20:00:00.000000",
  "service_version": "1.0.0"
}
```

#### 3. Interactive API Documentation
Open http://localhost:8000/docs in your browser to:
- See all available endpoints
- Try requests interactively
- View request/response schemas

## 🧠 How It Works

### Classification Workflow
```
1. Email arrives at API Gateway
        ↓
2. Validates input with Pydantic
        ↓
3. Forwards to Classifier Service
        ↓
4. LangGraph Workflow:
   ┌─────────────────┐
   │ Classify Node   │ → Analyzes email with LLM
   └────────┬────────┘
            │
   ┌────────▼─────────┐
   │ Check Confidence │
   └────────┬─────────┘
            │
      ┌─────┴─────┐
      │           │
   High (>0.75)  Low (<0.50)
      │           │
      ↓           ↓
   Return    Re-analyze Node → Enhanced analysis
   Result         │
                  ↓
              Return Result
        ↓
5. Returns classification to Gateway
        ↓
6. Gateway formats response
        ↓
7. Returns to client
```

### Key Design Patterns

**1. Microservices**: Each service has single responsibility
**2. Dependency Injection**: Easy testing and mocking
**3. Factory Pattern**: Provider-agnostic LLM configuration
**4. State Machine**: LangGraph manages workflow state
**5. Async/Await**: Non-blocking I/O throughout

## 🔧 Configuration

All configuration via environment variables:
```bash
# LLM Provider
LLM_PROVIDER=gemini  # Options: gemini, openai, anthropic
GEMINI_API_KEY=your-key-here

# Model Settings
MODEL_NAME=gemini-2.0-flash
MAX_TOKENS=1000
TEMPERATURE=0.0  # Deterministic classification

# Classification Thresholds
HIGH_CONFIDENCE_THRESHOLD=0.75
LOW_CONFIDENCE_THRESHOLD=0.50

# Logging
LOG_LEVEL=INFO
```

## 🧪 Development

### Run Individual Services
```bash
# API Gateway only
docker-compose up api-gateway

# Classifier only
docker-compose up classifier

# View logs
docker-compose logs -f api-gateway
docker-compose logs -f classifier
```

### Rebuild After Code Changes
```bash
# Rebuild specific service
docker-compose build api-gateway

# Rebuild all
docker-compose build

# Restart services
docker-compose restart
```

### Debug Inside Container
```bash
# Enter api-gateway container
docker-compose exec api-gateway bash

# Test internal communication
curl http://classifier:8001/health

# Check Python imports
python -c "from models.schemas import EmailInput; print('OK')"
```

## 📊 API Endpoints

### API Gateway (Port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service information |
| GET | `/health` | Simple health check |
| GET | `/api/v1/health` | Full system health (all services) |
| POST | `/api/v1/classify` | Classify an email |
| GET | `/docs` | Interactive API documentation |

### Classifier Service (Internal Port 8001)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service information |
| GET | `/health` | Health check |
| POST | `/classify` | Internal classification endpoint |

## 🐛 Troubleshooting

### Issue: Port already in use
```bash
# Windows
netstat -ano | findstr :8000

# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use 8001 on host instead
```

### Issue: Services can't communicate
```bash
# Check network
docker network ls
docker network inspect email-intelligence-network

# Restart services
docker-compose restart
```

### Issue: Module not found errors
```bash
# Check volumes are mounted
docker-compose config

# Rebuild with no cache
docker-compose build --no-cache
```

## 📝 License

MIT

## 👤 Author

**Juan Jose Aguado**
- GitHub: [@Aguado4](https://github.com/Aguado4)
- Project: [email-intelligence-system](https://github.com/Aguado4/email-intelligence-system)

---

**Status**: 🟢 Core microservices operational | 🔨 RAG enhancement in progress
