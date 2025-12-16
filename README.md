# Email Intelligence System

A production-ready microservices architecture for intelligent email classification using LangGraph and FastAPI.

## 🏗️ Architecture
```
┌─────────────┐      ┌──────────────────┐      ┌─────────────┐
│ API Gateway │─────▶│ Classifier       │─────▶│ PostgreSQL  │
│  (FastAPI)  │      │  (LangGraph)     │      │  Database   │
└─────────────┘      └──────────────────┘      └─────────────┘
       │                      │
       │                      │
       └──────────┬───────────┘
                  ▼
         ┌─────────────────┐
         │   Evaluator     │
         │   Service       │
         └─────────────────┘
```

## 🚀 Features

- **Async FastAPI Gateway**: High-performance async I/O
- **LangGraph Workflows**: Stateful, conditional routing
- **PostgreSQL Persistence**: Store classifications and metrics
- **Automated Evaluation**: F1, precision, recall metrics
- **Docker Compose**: One-command orchestration
- **Type Safety**: Pydantic models throughout

## 🛠️ Tech Stack

- **Python 3.11+**
- **FastAPI**: Modern async web framework
- **LangGraph**: LLM workflow orchestration
- **LangChain**: LLM integrations
- **PostgreSQL**: Relational database
- **Docker**: Containerization
- **Pytest**: Testing framework

## 📦 Project Status

🚧 **Work in Progress** - Building microservices incrementally

### Completed
- [x] Project structure
- [x] Shared Pydantic models
- [ ] Classifier service (LangGraph)
- [ ] API Gateway (FastAPI)
- [ ] Evaluator service
- [ ] Database integration
- [ ] Docker orchestration

## 🏃 Quick Start (Coming Soon)
```bash
# Clone repository
git clone https://github.com/Aguado4/email-intelligence-system.git

# Start all services
docker-compose up

# Access API docs
open http://localhost:8000/docs
```

## 📝 License

MIT