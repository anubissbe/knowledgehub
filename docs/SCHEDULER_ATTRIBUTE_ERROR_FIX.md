# Scheduler AttributeError Fix Documentation

## Problem Statement
The KnowledgeHub scheduler container was failing with an AttributeError when trying to access the `next_run_time` attribute of an APScheduler job object. The error occurred at line 151 in `/opt/projects/knowledgehub/src/scheduler/main.py`.

Error message:
```
AttributeError: 'Job' object has no attribute 'next_run_time'. Did you mean: '_get_run_times'?
```

## Root Cause
In APScheduler 3.10.4, the `next_run_time` attribute is not immediately available on a job object. It's only calculated and populated after:
1. The scheduler has been started
2. The job's trigger has been processed
3. The job store has been properly initialized

The code was trying to access this attribute immediately after adding the job but before these initialization steps were complete.

## Solution
Added proper attribute checking before accessing `next_run_time`:

```python
# Before (causing error):
job = self.scheduler.get_job("weekly_refresh")
if job:
    logger.info(f"Next refresh scheduled for: {job.next_run_time}")

# After (fixed):
job = self.scheduler.get_job("weekly_refresh")
if job and hasattr(job, 'next_run_time') and job.next_run_time:
    logger.info(f"Next refresh scheduled for: {job.next_run_time}")
elif job:
    logger.info("Weekly refresh job scheduled (next run time will be available soon)")
```

## Implementation Details

### File Modified
`/opt/projects/knowledgehub/src/scheduler/main.py` (lines 153-156)

### Changes Made
1. Added `hasattr()` check to verify the attribute exists
2. Added null check to ensure the value is available
3. Added fallback message when the attribute isn't ready yet

### Why This Works
- The `hasattr()` check prevents the AttributeError
- The additional null check ensures we don't log "None"
- The fallback message confirms the job was scheduled successfully
- APScheduler will populate `next_run_time` shortly after the scheduler starts

## Testing
After applying the fix:
1. Rebuilt the scheduler container
2. Restarted the container
3. Verified logs show successful startup without errors
4. Confirmed the next run time is logged: "Next refresh scheduled for: 2025-07-07 02:00:00+00:00"

## Additional Notes
- The scheduler runs a weekly refresh job every Sunday at 2:00 AM UTC
- The cron schedule is configurable via the `REFRESH_SCHEDULE` environment variable
- Default schedule: "0 2 * * 0" (2 AM every Sunday)
- The scheduler gracefully handles the attribute availability timing issue

## Prevention
For future APScheduler usage:
1. Always check attribute existence before accessing dynamic attributes
2. Consider using `job.get_next_fire_time()` method instead of direct attribute access
3. Be aware that job attributes may not be immediately available after job creation