"""
Property-Based Tests for Job Queue Manager

Feature: turkish-sentence-emotion-analysis, Property 28: FIFO Queue Ordering
Validates: Requirements 8.3
"""

import pytest
from hypothesis import given, strategies as st, settings
from sentence_analysis.job_queue import JobQueueManager, JobStatusEnum


@settings(max_examples=100)
@given(
    num_jobs=st.integers(min_value=2, max_value=20),
    client_ips=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=('Nd', 'Lu', 'Ll')),
            min_size=7,
            max_size=15
        ),
        min_size=1,
        max_size=5
    )
)
def test_fifo_queue_ordering(num_jobs, client_ips):
    """
    **Feature: turkish-sentence-emotion-analysis, Property 28: FIFO Queue Ordering**
    **Validates: Requirements 8.3**
    
    Property: For any sequence of queued requests, they should be processed 
    in the order they were received (first-in, first-out).
    
    This test verifies that:
    1. Jobs are added to the queue in the order they are created
    2. Jobs are retrieved from the queue in FIFO order
    3. The queue position reflects the correct ordering
    4. Processing jobs maintains FIFO order
    """
    # Create job queue manager with sufficient capacity
    queue_manager = JobQueueManager(
        max_queue_size=50,
        rate_limit_requests=100,  # High limit to avoid rate limiting in test
        rate_limit_window=60
    )
    
    # Create multiple jobs
    job_ids = []
    for i in range(num_jobs):
        # Use different client IPs to avoid rate limiting
        client_ip = client_ips[i % len(client_ips)]
        job_id = queue_manager.create_job(client_ip=client_ip)
        job_ids.append(job_id)
    
    # Verify all jobs are in queue
    assert queue_manager.get_queue_size() == num_jobs, \
        f"Expected {num_jobs} jobs in queue, got {queue_manager.get_queue_size()}"
    
    # Verify FIFO ordering: queue positions should match creation order
    for i, job_id in enumerate(job_ids):
        position = queue_manager.get_queue_position(job_id)
        assert position == i, \
            f"Job {i} (ID: {job_id}) should be at position {i}, but is at position {position}"
    
    # Verify jobs are retrieved in FIFO order
    retrieved_order = []
    for _ in range(num_jobs):
        next_job = queue_manager.get_next_job()
        assert next_job is not None, "Queue should not be empty"
        retrieved_order.append(next_job)
        
        # Start processing to remove from queue
        success = queue_manager.start_processing(next_job)
        assert success, f"Failed to start processing job {next_job}"
    
    # Verify retrieved order matches creation order
    assert retrieved_order == job_ids, \
        f"Jobs were not retrieved in FIFO order. Expected: {job_ids}, Got: {retrieved_order}"
    
    # Verify queue is now empty
    assert queue_manager.get_queue_size() == 0, \
        f"Queue should be empty after processing all jobs, but has {queue_manager.get_queue_size()} jobs"
    
    # Verify all jobs are in PROCESSING state
    for job_id in job_ids:
        job_status = queue_manager.get_job_status(job_id)
        assert job_status is not None, f"Job {job_id} not found"
        assert job_status.status == JobStatusEnum.PROCESSING, \
            f"Job {job_id} should be in PROCESSING state, but is in {job_status.status}"


@settings(max_examples=100)
@given(
    num_jobs=st.integers(min_value=3, max_value=15),
    num_to_process=st.integers(min_value=1, max_value=10)
)
def test_fifo_ordering_with_partial_processing(num_jobs, num_to_process):
    """
    Property: FIFO ordering is maintained even when only some jobs are processed.
    
    This test verifies that:
    1. Jobs are always processed from the front of the queue (FIFO)
    2. Remaining jobs maintain FIFO order after partial processing
    3. Queue positions update correctly after processing
    """
    # Limit num_to_process to be less than num_jobs
    num_to_process = min(num_to_process, num_jobs - 1)
    if num_to_process < 1:
        num_to_process = 1
    
    # Create job queue manager
    queue_manager = JobQueueManager(
        max_queue_size=50,
        rate_limit_requests=100,
        rate_limit_window=60
    )
    
    # Create jobs
    job_ids = []
    for i in range(num_jobs):
        job_id = queue_manager.create_job(client_ip=f"192.168.1.{i}")
        job_ids.append(job_id)
    
    # Process first num_to_process jobs in FIFO order
    processed_jobs = []
    for i in range(num_to_process):
        # Get next job (should always be the first in queue)
        next_job = queue_manager.get_next_job()
        
        # Should match the expected job in FIFO order
        assert next_job == job_ids[i], \
            f"Expected job at index {i} to be next, but got different job"
        
        queue_manager.start_processing(next_job)
        processed_jobs.append(next_job)
    
    # Verify remaining jobs are still in FIFO order
    remaining_jobs = job_ids[num_to_process:]
    
    for i, job_id in enumerate(remaining_jobs):
        position = queue_manager.get_queue_position(job_id)
        assert position == i, \
            f"Remaining job should be at position {i}, but is at position {position}"


@settings(max_examples=100)
@given(
    num_jobs=st.integers(min_value=1, max_value=10)
)
def test_fifo_ordering_single_client(num_jobs):
    """
    Property: FIFO ordering is maintained for jobs from a single client.
    
    This test verifies that multiple jobs from the same client are processed
    in the order they were submitted.
    """
    queue_manager = JobQueueManager(
        max_queue_size=50,
        rate_limit_requests=100,
        rate_limit_window=60
    )
    
    client_ip = "192.168.1.100"
    job_ids = []
    
    # Create multiple jobs from same client
    for _ in range(num_jobs):
        job_id = queue_manager.create_job(client_ip=client_ip)
        job_ids.append(job_id)
    
    # Verify FIFO ordering
    for i in range(num_jobs):
        next_job = queue_manager.get_next_job()
        assert next_job == job_ids[i], \
            f"Expected job {job_ids[i]} at position {i}, but got {next_job}"
        
        queue_manager.start_processing(next_job)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
