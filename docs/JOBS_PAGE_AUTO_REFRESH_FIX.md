# Jobs Page Auto-Refresh Fix Documentation

## Problem Statement
The Jobs page at http://192.168.1.25:3101/jobs was not automatically refreshing when new jobs were created or job statuses changed. Users had to manually refresh the browser to see updates.

## Solution Overview
Implemented auto-refresh functionality for the Jobs page similar to the Dashboard, with:
1. Automatic refresh every 5 seconds
2. Manual refresh button
3. WebSocket notifications (already implemented)
4. Cache invalidation on job actions (already implemented)

## Implementation Details

### Frontend Updates

**File: `/src/web-ui/src/pages/Jobs.tsx`**

1. **Added Auto-refresh Interval**:
```typescript
const { data: jobsResponse, isLoading, refetch, isRefetching } = useQuery({
  queryKey: ['jobs', page, rowsPerPage],
  queryFn: () => api.getJobs(),
  refetchInterval: 5000, // Refresh every 5 seconds for more frequent job updates
  refetchIntervalInBackground: true, // Continue refreshing even when tab is not active
})
```

2. **Added Manual Refresh Button**:
```typescript
<Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
  <Typography variant="h4" sx={{ flexGrow: 1 }}>
    Jobs
  </Typography>
  <Tooltip title="Refresh jobs">
    <IconButton 
      onClick={() => refetch()} 
      disabled={isRefetching}
      color="primary"
    >
      <SyncIcon />
    </IconButton>
  </Tooltip>
</Box>
```

3. **Import Updates**:
- Added `Tooltip` component for better UX
- Added `Sync as SyncIcon` for the refresh button
- Added `isRefetching` to disable button during refresh

### How It Works

1. **Automatic Updates**:
   - Jobs list refreshes every 5 seconds automatically
   - More frequent than dashboard (5s vs 30s) due to jobs being more time-sensitive
   - Continues refreshing even when browser tab is not active

2. **Manual Refresh**:
   - Users can click the sync icon for immediate refresh
   - Button is disabled while refresh is in progress
   - Provides visual feedback with tooltip

3. **Existing Features Still Working**:
   - WebSocket notifications trigger immediate updates when jobs complete
   - Cache invalidation when users retry/cancel jobs
   - Query invalidation cascades to dashboard stats

## Complete Update Flow

1. **New Job Created** (from any source):
   - Jobs page auto-refreshes within 5 seconds
   - User sees new job appear in the list

2. **Job Status Changes**:
   - Backend sends WebSocket notification
   - Frontend receives notification and invalidates jobs query
   - Jobs list updates immediately
   - If WebSocket fails, auto-refresh catches it within 5 seconds

3. **User Actions** (retry/cancel):
   - Action triggers immediate cache invalidation
   - Jobs list updates instantly
   - Dashboard stats also update

## Benefits

1. **No Manual Refresh Needed**: Jobs list stays current automatically
2. **Multiple Update Paths**: Ensures reliability even if one method fails
3. **Responsive UI**: Users see changes within seconds
4. **Background Updates**: Works even when user is on another tab
5. **Visual Feedback**: Manual refresh button shows when updates are happening

## Testing

To verify the implementation:

1. Open http://192.168.1.25:3101/jobs
2. Create a new job from another source or tab
3. Within 5 seconds, the new job should appear
4. Click the sync button to force immediate refresh
5. Watch job status changes update automatically
6. Leave the tab and come back - updates should continue

## Performance Considerations

- 5-second interval is aggressive but necessary for job monitoring
- React Query deduplicates requests if multiple triggers occur
- Background refetching can be disabled if performance is a concern
- Consider increasing interval if server load becomes an issue