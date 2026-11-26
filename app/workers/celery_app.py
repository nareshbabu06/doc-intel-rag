from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "doc_intel_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
    result_expires=86400,
)
