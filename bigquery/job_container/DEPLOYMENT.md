# Deployment Guide: Resilient BigQuery Sync Job

## Quick Deploy

```bash
# Deploy the new resilient job
gcloud builds submit --config=bigquery/job_container/cloudbuild.yaml
```

## What This Fixes

### Before (Old Job)
- ❌ Fails completely when hitting large blog post
- ❌ No items processed if one fails
- ❌ Exits with error code 1
- ❌ No batch sizing for token limits

### After (New Job)
- ✅ Processes all items it can
- ✅ Skips problematic items gracefully
- ✅ Exits successfully if any items processed
- ✅ Automatic batch sizing handles large posts

## Verification Steps

After deployment:

1. **Check job status**:
   ```bash
   gcloud run jobs describe bigquery-sync-job --region=europe-west3
   ```

2. **Test run manually**:
   ```bash
   gcloud run jobs execute bigquery-sync-job --region=europe-west3
   ```

3. **Monitor logs**:
   ```bash
   gcloud logging read 'resource.type=cloud_run_job AND resource.labels.job_name=bigquery-sync-job' --limit=50
   ```

4. **Verify success**:
   - Look for "Job completed with partial/full success"
   - Check that some items were processed even if others failed
   - Verify no "CRITICAL: No items processed" errors

## Expected Behavior

### Normal Operation
- Processes new blog posts from GCS
- Creates batches that respect token limits
- Stores items and chunks in BigQuery
- Exits with code 0 (success)

### With Large Posts
- Detects large post (>200KB)
- Uses smaller chunks (750 or 500 tokens)
- Splits into multiple batches
- Processes successfully

### With Failures
- Logs error for failed item
- Continues processing other items
- Reports partial success
- Exits with code 0 if any items succeeded

## Rollback

If issues occur:

```bash
# Revert to previous image
gcloud run jobs update bigquery-sync-job \
  --image=europe-west3-docker.pkg.dev/nvidia-blog/bigquery-sync/bigquery-sync-job:previous \
  --region=europe-west3
```

## Monitoring

Key metrics to watch:
- Success rate: Should be > 0 even with problematic posts
- Batch creation: Should see multiple batches for large posts
- Error logs: Should show clear reasons for any failures
- Processing time: Should be reasonable (< 1 hour)
