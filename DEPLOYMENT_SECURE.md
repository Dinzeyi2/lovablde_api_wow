# 🔒 ALGOAPI SECURE - PRODUCTION DEPLOYMENT GUIDE

## 🛡️ What's Different (Secure vs Original)

### Security Improvements:
- ✅ **Docker Isolation** - All code runs in containers
- ✅ **Celery Task Queue** - Training jobs don't block web server
- ✅ **Rate Limiting** - Prevents abuse (slowapi)
- ✅ **Input Validation** - Pydantic schemas for all inputs
- ✅ **Data Sanitizer** - Auto-clean dirty data
- ✅ **Logic Verification** - Prove algorithm correctness
- ✅ **No Custom Code Upload** - Only pre-built algorithms (prevents RCE)

### Architecture Changes:
```
OLD: FastAPI → Direct Execution → Response
NEW: FastAPI → Redis → Celery Worker → Response
```

---

## 🚀 DEPLOYMENT OPTIONS

### Option A: Railway (Easiest - 10 minutes)

#### Step 1: Prerequisites
```bash
# Install Docker (required for local testing)
docker --version  # Should show version

# Install Redis locally (for testing)
brew install redis  # Mac
# OR
apt-get install redis-server  # Linux
```

#### Step 2: Deploy to Railway

1. **Push to GitHub**
```bash
cd algoapi-secure
git init
git add .
git commit -m "AlgoAPI Secure v2.0"
git remote add origin YOUR_REPO
git push -u origin main
```

2. **Create Railway Project**
- Go to railway.app
- New Project → Deploy from GitHub
- Select your repo

3. **Add Services**
```
Service 1: Web (FastAPI)
- Auto-detected from requirements.txt
- Add environment variable: REDIS_URL=${{Redis.REDIS_URL}}

Service 2: PostgreSQL
- Click "New" → Database → PostgreSQL
- Automatically sets DATABASE_URL

Service 3: Redis
- Click "New" → Database → Redis
- Automatically sets REDIS_URL

Service 4: Celery Worker
- Click "New" → Empty Service
- Connect same GitHub repo
- Custom Start Command: celery -A app.celery_config worker --loglevel=info
- Add environment variable: REDIS_URL=${{Redis.REDIS_URL}}
- Add environment variable: DATABASE_URL=${{Postgres.DATABASE_URL}}
```

4. **Generate Domain**
- Go to Web service → Settings → Generate Domain
- Your API: `https://your-project.railway.app`

---

### Option B: AWS (Production Scale)

#### Components:
1. **ECS Fargate** - Run FastAPI containers
2. **ElastiCache Redis** - Task queue
3. **RDS PostgreSQL** - Database
4. **ECS Fargate** - Celery workers (separate service)
5. **ALB** - Load balancer
6. **ECR** - Docker images

Cost: ~$50-100/month for low traffic

---

### Option C: DigitalOcean App Platform

1. **Create App**
- Deploy from GitHub
- Add Redis managed database
- Add PostgreSQL managed database

2. **Add Worker Component**
```yaml
# .do/app.yaml
name: algoapi-secure
services:
  - name: web
    source:
      repo: your-repo
    run_command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
  - name: worker
    source:
      repo: your-repo
    run_command: celery -A app.celery_config worker --loglevel=info

databases:
  - name: postgres
  - name: redis
```

Cost: ~$30/month

---

## 🔧 LOCAL DEVELOPMENT

### Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Redis
redis-server

# 3. Start PostgreSQL (or use SQLite for testing)
# export DATABASE_URL=sqlite:///./algoapi.db

# 4. Start Celery worker (separate terminal)
celery -A app.celery_config worker --loglevel=info

# 5. Start FastAPI
uvicorn app.main:app --reload
```

### Test

```bash
# Health check
curl http://localhost:8000/health

# Should return:
# {
#   "status": "healthy",
#   "version": "2.0.0-secure",
#   "docker_available": true,
#   "isolation": "enabled"
# }
```

---

## 🧪 TESTING THE SECURE VERSION

### 1. Test Rate Limiting

```bash
# Try creating 10 API keys rapidly
for i in {1..10}; do
  curl -X POST http://localhost:8000/api/v1/keys/create \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"Test$i\",\"email\":\"test$i@example.com\"}"
done

# After 5 requests, you should get:
# {"detail":"Rate limit exceeded: 5 per 1 hour"}
```

### 2. Test Background Training

```bash
# Upload CSV and train
curl -X POST http://localhost:8000/api/v1/train \
  -H "X-API-Key: your-key" \
  -F "file=@data.csv" \
  -F "model_type=recommendation"

# Returns task_id immediately (doesn't block)
# Check Celery worker logs to see training progress
```

### 3. Test Data Sanitizer

Upload a CSV with issues:
```csv
user_id,item_id,rating,bad_column
1,A,5,
1,B,,extra
,C,4,data
```

The sanitizer will:
- Fill missing values
- Remove problematic columns
- Return data quality report

### 4. Test Docker Isolation

```bash
# Check isolation status
curl http://localhost:8000/health

# Should show: "docker_available": true
```

---

## 📊 MONITORING

### Celery Flower (Task Monitor)

```bash
# Install
pip install flower

# Run
celery -A app.celery_config flower

# Access: http://localhost:5555
```

### Health Checks

```bash
# Web service
curl https://your-app.railway.app/health

# Celery workers (via Flower)
curl http://localhost:5555/api/workers
```

---

## 🔐 SECURITY CHECKLIST

### Before Production:

- [ ] Docker installed and running
- [ ] Redis configured (Railway provides this)
- [ ] PostgreSQL configured
- [ ] Rate limits tested
- [ ] Environment variables set
- [ ] HTTPS enabled (Railway default)
- [ ] API keys generated
- [ ] Celery workers running
- [ ] Data sanitizer tested
- [ ] Error handling tested

### Optional Enhancements:

- [ ] Add Sentry for error tracking
- [ ] Set up log aggregation (Papertrail)
- [ ] Configure auto-scaling
- [ ] Add metrics (Prometheus)
- [ ] Set up backups

---

## 💰 COST ESTIMATE

### Railway (Recommended for MVP):
- **Starter Plan**: $5/month
  - Web service: Included
  - PostgreSQL: Included
  - Redis: Included
  - Celery worker: Included
- **Scale as needed**

### AWS (Production Scale):
- **Small**: ~$80/month
  - Fargate tasks (2)
  - RDS PostgreSQL
  - ElastiCache Redis
  - Load Balancer
- **Medium**: ~$300/month (more workers, bigger DB)

---

## 🆘 TROUBLESHOOTING

### Celery worker not processing tasks

```bash
# Check Redis connection
redis-cli ping
# Should return: PONG

# Check Celery can connect
celery -A app.celery_config inspect active

# Check task queue
redis-cli
> KEYS celery*
```

### Docker not available

```bash
# Install Docker
# Mac: Download Docker Desktop
# Linux: sudo apt-get install docker.io

# Start Docker daemon
sudo systemctl start docker

# Verify
docker ps
```

### Training jobs timeout

```bash
# Increase timeout in tasks.py
task_time_limit=1200  # 20 minutes instead of 10
```

---

## 📈 SCALING GUIDE

### When to Scale:

- **10-100 users**: Single Railway instance OK
- **100-1000 users**: Add more Celery workers
- **1000+ users**: Move to AWS/GCP with auto-scaling

### How to Scale:

```bash
# Railway: Increase worker count
# Settings → Replicas → 3

# AWS: Auto-scaling group
aws ecs update-service --service celery-workers --desired-count 5
```

---

## 🎯 PRODUCTION READINESS

### ✅ What's Ready:
- Security (Docker, rate limiting)
- Task queue (Celery)
- Data quality (Sanitizer)
- Input validation
- Error handling

### 🚧 What's Next (Month 2):
- Custom metrics
- User dashboard
- Billing integration (Stripe)
- Model versioning
- A/B testing

---

## 📝 ENVIRONMENT VARIABLES

Required:
```bash
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379/0
PORT=8000
```

Optional:
```bash
SENTRY_DSN=your-sentry-dsn
MAX_WORKERS=4
LOG_LEVEL=info
```

---

**Your secure, production-ready API is ready to deploy! 🚀**

Choose Railway for easy start, AWS for scale.
