"""
Tests for progress tracking service.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.progress_tracker import ProgressTracker, TaskStatus

client = TestClient(app)


class TestTaskStatus:
    """Tests for TaskStatus"""
    
    def test_status_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.PROCESSING == "processing"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"


class TestProgressTracker:
    """Tests for ProgressTracker"""
    
    @pytest.fixture
    def tracker(self):
        return ProgressTracker()
    
    def test_create_task(self, tracker):
        task_id = tracker.create_task("test_task")
        status = tracker.get_status(task_id)
        
        assert status["status"] == TaskStatus.PENDING
        assert status["progress"] == 0
        assert status["total"] == 100
    
    def test_update_progress(self, tracker):
        task_id = tracker.create_task("test_task")
        
        tracker.update(task_id, progress=50, message="Half done")
        status = tracker.get_status(task_id)
        
        assert status["progress"] == 50
        assert status["message"] == "Half done"
    
    def test_complete_task(self, tracker):
        task_id = tracker.create_task("test_task")
        
        tracker.complete(task_id)
        status = tracker.get_status(task_id)
        
        assert status["status"] == TaskStatus.COMPLETED
        assert status["progress"] == 100
    
    def test_fail_task(self, tracker):
        task_id = tracker.create_task("test_task")
        
        tracker.fail(task_id, error="Something went wrong")
        status = tracker.get_status(task_id)
        
        assert status["status"] == TaskStatus.FAILED
        assert status["error"] == "Something went wrong"
    
    def test_nonexistent_task(self, tracker):
        status = tracker.get_status("nonexistent")
        assert status is None


class TestProgressAPI:
    """Tests for progress SSE endpoint"""
    
    def test_progress_endpoint_exists(self):
        """Test that progress endpoint is registered (404 = no such job, not 404 = no route)"""
        response = client.get("/job/test-id/progress")
        # 404 means route exists but job not found — that's correct
        # If route didn't exist, FastAPI would return 404 differently
        assert response.status_code == 404
        assert response.json()["detail"] == "Task not found"
    
    def test_progress_nonexistent_job(self):
        response = client.get("/job/nonexistent-id/progress")
        assert response.status_code == 404