"""
Job Queue Manager Module

Handles asynchronous processing of sentence analysis requests with FIFO ordering.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
from datetime import datetime
from enum import Enum
import uuid
import asyncio
from collections import deque
import time


class JobStatusEnum(str, Enum):
    """Job status enumeration"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class JobStatus:
    """Job status information"""
    job_id: str
    status: JobStatusEnum
    progress: float  # 0.0 to 1.0
    result: Optional[Dict] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    processing_time: Optional[float] = None


class JobQueueManager:
    """Manages job queue with FIFO ordering and rate limiting"""
    
    def __init__(
        self,
        max_queue_size: int = 50,
        rate_limit_requests: int = 10,
        rate_limit_window: int = 60,  # seconds
        processing_timeout: int = 30  # seconds
    ):
        """
        Initialize JobQueueManager
        
        Args:
            max_queue_size: Maximum number of queued jobs
            rate_limit_requests: Maximum requests per window
            rate_limit_window: Time window for rate limiting in seconds
            processing_timeout: Maximum processing time per job in seconds
        """
        self.max_queue_size = max_queue_size
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window = rate_limit_window
        self.processing_timeout = processing_timeout
        
        # FIFO queue for job IDs
        self.job_queue: deque = deque()
        
        # Job storage
        self.jobs: Dict[str, JobStatus] = {}
        
        # Rate limiting tracking
        self.client_requests: Dict[str, list] = {}
        
        # Lock for thread-safe operations
        self.lock = asyncio.Lock()
    
    def create_job(self, client_ip: str = "unknown") -> str:
        """
        Create a new job and add to queue
        
        Args:
            client_ip: Client IP address for rate limiting
            
        Returns:
            Job ID
            
        Raises:
            ValueError: If queue is full or rate limit exceeded
        """
        # Check rate limiting
        if not self.check_rate_limit(client_ip):
            raise ValueError(
                f"Rate limit exceeded: {self.rate_limit_requests} requests per "
                f"{self.rate_limit_window} seconds"
            )
        
        # Check queue size
        if len(self.job_queue) >= self.max_queue_size:
            raise ValueError(f"Queue is full: {self.max_queue_size} jobs pending")
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create job status
        job_status = JobStatus(
            job_id=job_id,
            status=JobStatusEnum.QUEUED,
            progress=0.0,
            created_at=datetime.now()
        )
        
        # Add to queue (FIFO)
        self.job_queue.append(job_id)
        self.jobs[job_id] = job_status
        
        # Track request for rate limiting
        self._track_request(client_ip)
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """
        Get status of a job
        
        Args:
            job_id: Job identifier
            
        Returns:
            JobStatus object or None if not found
        """
        return self.jobs.get(job_id)
    
    def get_next_job(self) -> Optional[str]:
        """
        Get next job from queue (FIFO)
        
        Returns:
            Job ID or None if queue is empty
        """
        if self.job_queue:
            return self.job_queue[0]
        return None
    
    def start_processing(self, job_id: str) -> bool:
        """
        Mark job as processing and remove from queue
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if successful, False if job not found
        """
        if job_id not in self.jobs:
            return False
        
        # Remove from queue
        if job_id in self.job_queue:
            self.job_queue.remove(job_id)
        
        # Update status
        self.jobs[job_id].status = JobStatusEnum.PROCESSING
        self.jobs[job_id].progress = 0.1
        
        return True
    
    def update_progress(self, job_id: str, progress: float) -> bool:
        """
        Update job progress
        
        Args:
            job_id: Job identifier
            progress: Progress value (0.0 to 1.0)
            
        Returns:
            True if successful, False if job not found
        """
        if job_id not in self.jobs:
            return False
        
        self.jobs[job_id].progress = max(0.0, min(1.0, progress))
        return True
    
    def complete_job(self, job_id: str, result: Dict) -> bool:
        """
        Mark job as completed with result
        
        Args:
            job_id: Job identifier
            result: Analysis result dictionary
            
        Returns:
            True if successful, False if job not found
        """
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        job.status = JobStatusEnum.COMPLETED
        job.progress = 1.0
        job.result = result
        job.completed_at = datetime.now()
        
        # Calculate processing time
        if job.created_at:
            job.processing_time = (job.completed_at - job.created_at).total_seconds()
        
        return True
    
    def fail_job(self, job_id: str, error: str) -> bool:
        """
        Mark job as failed with error message
        
        Args:
            job_id: Job identifier
            error: Error message
            
        Returns:
            True if successful, False if job not found
        """
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        job.status = JobStatusEnum.FAILED
        job.error = error
        job.completed_at = datetime.now()
        
        # Calculate processing time
        if job.created_at:
            job.processing_time = (job.completed_at - job.created_at).total_seconds()
        
        return True
    
    def timeout_job(self, job_id: str) -> bool:
        """
        Mark job as timed out
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if successful, False if job not found
        """
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        job.status = JobStatusEnum.TIMEOUT
        job.error = f"Processing exceeded {self.processing_timeout} seconds timeout"
        job.completed_at = datetime.now()
        
        if job.created_at:
            job.processing_time = (job.completed_at - job.created_at).total_seconds()
        
        return True
    
    def check_rate_limit(self, client_ip: str) -> bool:
        """
        Check if client has exceeded rate limit
        
        Args:
            client_ip: Client IP address
            
        Returns:
            True if within rate limit, False if exceeded
        """
        current_time = time.time()
        
        # Clean up old requests
        if client_ip in self.client_requests:
            self.client_requests[client_ip] = [
                req_time for req_time in self.client_requests[client_ip]
                if current_time - req_time < self.rate_limit_window
            ]
        else:
            self.client_requests[client_ip] = []
        
        # Check if limit exceeded
        return len(self.client_requests[client_ip]) < self.rate_limit_requests
    
    def _track_request(self, client_ip: str):
        """
        Track request for rate limiting
        
        Args:
            client_ip: Client IP address
        """
        current_time = time.time()
        
        if client_ip not in self.client_requests:
            self.client_requests[client_ip] = []
        
        self.client_requests[client_ip].append(current_time)
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return len(self.job_queue)
    
    def get_queue_position(self, job_id: str) -> Optional[int]:
        """
        Get position of job in queue
        
        Args:
            job_id: Job identifier
            
        Returns:
            Queue position (0-indexed) or None if not in queue
        """
        try:
            return list(self.job_queue).index(job_id)
        except ValueError:
            return None
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """
        Remove old completed/failed jobs from storage
        
        Args:
            max_age_hours: Maximum age in hours to keep jobs
        """
        current_time = datetime.now()
        jobs_to_remove = []
        
        for job_id, job in self.jobs.items():
            if job.completed_at:
                age_hours = (current_time - job.completed_at).total_seconds() / 3600
                if age_hours > max_age_hours:
                    jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.jobs[job_id]
