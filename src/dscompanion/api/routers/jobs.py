"""Background-job polling endpoint, shared by Step 8 (leaderboard) and Step 9 (tuning)
once they're ported in Phase G — the REST equivalent of Step 8's
``st.fragment(run_every=1)`` polling loop.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from dscompanion.api.deps import get_auth, get_pipeline_service
from dscompanion.api.schemas import JobStatusResponse
from dscompanion.api.state import RunState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/runs", tags=["jobs"], dependencies=[Depends(get_auth)])

__all__ = ["router"]


@router.get("/{run_id}/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, run: RunState = Depends(get_pipeline_service)) -> JobStatusResponse:
    """Returns a background job's current status and progress.

    Args:
        job_id (str): The job identifier returned when the job was started.
        run (RunState): Injected via ``get_pipeline_service`` from the ``run_id`` path
            parameter.

    Returns:
        JobStatusResponse: Current status, progress entries, and result/error once
        finished.

    Raises:
        HTTPException: 404, if ``job_id`` is not known for this run.
    """
    if job_id not in run.jobs:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    job = run.jobs[job_id]
    with job.lock:
        return JobStatusResponse(
            job_id=job.job_id,
            step=job.step,
            status=job.status,
            progress=list(job.progress),
            result=job.result,
            error=job.error,
        )
