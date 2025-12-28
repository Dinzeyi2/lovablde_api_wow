"""
Celery Worker Configuration for AlgoAPI
Handles all background tasks: model training, data processing, algorithm execution
"""

from celery import Celery
from celery.schedules import crontab
import os

# Redis connection (Railway will provide REDIS_URL)
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Create Celery app
celery_app = Celery(
    'algoapi',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['app.tasks']
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max per task
    task_soft_time_limit=540,  # 9 minutes soft limit
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks (prevent memory leaks)
    broker_connection_retry_on_startup=True,
)

# Periodic tasks (optional - for cleanup, monitoring)
celery_app.conf.beat_schedule = {
    'cleanup-old-models': {
        'task': 'app.tasks.cleanup_old_models',
        'schedule': crontab(hour=2, minute=0),  # 2 AM daily
    },
    'check-training-status': {
        'task': 'app.tasks.check_stalled_training',
        'schedule': 300.0,  # Every 5 minutes
    },
}

if __name__ == '__main__':
    celery_app.start()
