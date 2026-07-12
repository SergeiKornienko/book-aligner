"""
Progress tracking service for async tasks.
"""

import uuid
import asyncio
from typing import Dict, Optional
from backend.logger import logger


class TaskStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProgressTracker:
    """Tracks progress of async tasks."""
    
    def __init__(self):
        self._tasks: Dict[str, dict] = {}
        self._events: Dict[str, asyncio.Event] = {}
    
    def create_task(self, name: str, total: int = 100) -> str:
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = {
            "task_id": task_id,
            "name": name,
            "status": TaskStatus.PENDING,
            "progress": 0,
            "total": total,
            "message": "",
            "error": None
        }
        self._events[task_id] = asyncio.Event()
        logger.debug(f"Created task {task_id}: {name}")
        return task_id
    
    def update(self, task_id: str, progress: int = None, message: str = None):
        if task_id not in self._tasks:
            return
        
        task = self._tasks[task_id]
        task["status"] = TaskStatus.PROCESSING
        
        if progress is not None:
            task["progress"] = progress
        if message is not None:
            task["message"] = message
        
        # Notify listeners
        if task_id in self._events:
            self._events[task_id].set()
            self._events[task_id] = asyncio.Event()
        
        logger.debug(f"Task {task_id}: {task['progress']}% - {task['message']}")
    
    def complete(self, task_id: str):
        if task_id not in self._tasks:
            return
        
        self._tasks[task_id]["status"] = TaskStatus.COMPLETED
        self._tasks[task_id]["progress"] = 100
        self._tasks[task_id]["message"] = "Complete"
        
        if task_id in self._events:
            self._events[task_id].set()
        
        logger.info(f"Task {task_id} completed")
    
    def fail(self, task_id: str, error: str):
        if task_id not in self._tasks:
            return
        
        self._tasks[task_id]["status"] = TaskStatus.FAILED
        self._tasks[task_id]["error"] = error
        
        if task_id in self._events:
            self._events[task_id].set()
        
        logger.error(f"Task {task_id} failed: {error}")
    
    def get_status(self, task_id: str) -> Optional[dict]:
        return self._tasks.get(task_id)
    
    async def wait_for_update(self, task_id: str, timeout: float = 30):
        if task_id in self._events:
            try:
                await asyncio.wait_for(self._events[task_id].wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass


# Singleton instance
progress_tracker = ProgressTracker()