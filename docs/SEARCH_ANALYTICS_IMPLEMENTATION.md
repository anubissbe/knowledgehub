# Search Analytics Implementation

## Overview

The Search Analytics feature provides comprehensive monitoring and visualization of search performance in KnowledgeHub. It tracks all search queries, analyzes usage patterns, and presents real-time metrics through an interactive dashboard.

## Features

### 📊 Analytics Dashboard
- **Location**: `/analytics` page in the web interface
- **Real-time Updates**: Automatic refresh every 30 seconds
- **Interactive Charts**: Built with Recharts for responsive visualizations
- **Performance Metrics**: Search volume, response times, and success rates

### 📈 Metrics Tracked

#### Search Volume
- **Today**: Current day search count
- **Week**: Last 7 days search count  
- **Month**: Last 30 days search count

#### Performance Metrics
- **Average Response Time**: Mean query execution time in milliseconds
- **Success Rate**: Percentage of queries that returned results
- **Hourly Activity**: Searches performed in the last hour

#### Search Type Distribution
- **Hybrid**: Combined vector + keyword search
- **Vector**: Semantic similarity search
- **Keyword**: Traditional full-text search

#### Popular Queries
- **Top 10 Queries**: Most frequently searched terms
- **Frequency**: Number of times each query was executed
- **Average Results**: Mean number of results returned per query

#### Daily Performance Trends
- **7-Day History**: Search volume and performance over time
- **Response Time Trends**: Daily average response times
- **Result Quality**: Average results returned per day

#### Real-time Activity
- **Recent Queries**: Last 5 searches with response times
- **Live Metrics**: Updated every 30 seconds
- **Performance Status**: Color-coded response time indicators

## Technical Implementation

### Backend Components

#### Search History Tracking
```python
# Location: src/api/models/search.py
class SearchHistory(Base):
    __tablename__ = "search_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(Text, nullable=False)
    results_count = Column(Integer, default=0)
    search_type = Column(String(50), default="hybrid")
    filters = Column(JSON, default={})
    execution_time_ms = Column(Float)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
```

#### Analytics Endpoints
```python
# Location: src/api/routes/analytics.py

@router.get("/search")
async def get_search_analytics(db: Session = Depends(get_db)):
    """Get comprehensive search analytics"""
    # Returns search volume, performance, types, and trends

@router.get("/search/realtime") 
async def get_realtime_search_metrics(db: Session = Depends(get_db)):
    """Get real-time search metrics for dashboard"""
    # Returns hourly searches and recent query activity
```

#### Automatic Tracking
```python
# Location: src/api/services/search_service.py
async def _record_search_analytics(self, db: AsyncSession, query: SearchQuery, response: Dict[str, Any]):
    """Record search analytics for performance monitoring"""
    search_record = SearchHistory(
        query=query.query,
        results_count=response.get("total", 0),
        search_type=query.search_type.value,
        filters=query.filters or {},
        execution_time_ms=response.get("search_time_ms", 0)
    )
    db.add(search_record)
    await db.commit()
```

### Frontend Components

#### Analytics Page
```typescript
// Location: src/web-ui/src/pages/Analytics.tsx
const Analytics: React.FC = () => {
  const [searchAnalytics, setSearchAnalytics] = useState<any>(null)
  const [realtimeMetrics, setRealtimeMetrics] = useState<any>(null)
  
  // Automatic refresh every 30 seconds for real-time data
  useEffect(() => {
    const interval = setInterval(async () => {
      const realtimeData = await api.getRealtimeSearchMetrics()
      setRealtimeMetrics(realtimeData)
    }, 30000)
    
    return () => clearInterval(interval)
  }, [])
}
```

#### API Integration
```typescript
// Location: src/web-ui/src/services/api.ts
export const api = {
  getSearchAnalytics: async () => {
    const { data } = await apiClient.get('/api/v1/analytics/search')
    return data
  },
  
  getRealtimeSearchMetrics: async () => {
    const { data } = await apiClient.get('/api/v1/analytics/search/realtime')
    return data
  }
}
```

### Database Schema

The analytics system uses the existing `search_history` table:

```sql
CREATE TABLE search_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query TEXT NOT NULL,
    results_count INTEGER DEFAULT 0,
    search_type VARCHAR(50) DEFAULT 'hybrid',
    filters JSONB DEFAULT '{}',
    execution_time_ms FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes
CREATE INDEX idx_search_created_at ON search_history(created_at DESC);
CREATE INDEX idx_search_query ON search_history USING gin (to_tsvector('english', query));
```

## Usage Guide

### Accessing Analytics
1. Navigate to the KnowledgeHub web interface
2. Click **Analytics** in the sidebar navigation
3. View real-time metrics and historical trends

### Understanding Metrics

#### Search Volume Cards
- **Today's Searches**: Total searches performed today
- **Average Response Time**: Current system performance
- **Success Rate**: Percentage of successful queries
- **Hourly Searches**: Recent activity level

#### Performance Charts
- **Daily Performance**: Line chart showing search trends over 7 days
- **Search Types**: Pie chart showing distribution of search methods
- **Popular Queries**: Table of most frequent search terms
- **Recent Activity**: Live feed of recent search queries

#### Color Coding
- **Green**: Fast response times (<100ms)
- **Orange**: Moderate response times (100-200ms)  
- **Red**: Slow response times (>200ms)

### Monitoring Performance

#### Key Performance Indicators
- **Response Time**: Target <150ms average
- **Success Rate**: Target >90%
- **Search Volume**: Monitor for usage patterns
- **Error Rate**: Watch for failed queries

#### Performance Optimization
- Monitor popular queries for optimization opportunities
- Track response time trends to identify performance degradation
- Analyze search type effectiveness for query routing improvements
- Use analytics to guide caching and indexing strategies

## Troubleshooting

### Common Issues

#### No Data Appearing
1. **Check Database Connection**: Ensure PostgreSQL is accessible
2. **Verify Search Activity**: Perform test searches to generate data
3. **Check API Endpoints**: Test `/api/v1/analytics/search` directly
4. **Review Logs**: Check API logs for analytics recording errors

#### Slow Performance
1. **Database Indexes**: Ensure search_history table has proper indexes
2. **Query Optimization**: Review analytics SQL queries for efficiency
3. **Cache Settings**: Verify frontend caching is not stale
4. **Resource Usage**: Monitor CPU/memory during analytics queries

#### Incorrect Metrics
1. **Time Zone Issues**: Verify UTC timestamps in database
2. **Date Filters**: Check SQL date comparison logic
3. **Aggregation Logic**: Validate calculation formulas
4. **Data Integrity**: Ensure search recording is consistent

### Debugging Commands

```bash
# Check search history data
docker exec knowledgehub-postgres psql -U khuser -d knowledgehub -c "SELECT COUNT(*) FROM search_history;"

# Test analytics endpoint
curl -s http://localhost:3000/api/v1/analytics/search | jq .

# Check recent search activity
docker exec knowledgehub-postgres psql -U khuser -d knowledgehub -c "SELECT query, created_at FROM search_history ORDER BY created_at DESC LIMIT 5;"

# Verify API logs
docker logs knowledgehub-api --tail=50 | grep analytics
```

## Configuration

### Environment Variables
- **DATABASE_URL**: PostgreSQL connection string
- **REDIS_URL**: Redis connection for caching
- **LOG_LEVEL**: Set to DEBUG for detailed analytics logging

### Performance Tuning
```python
# src/api/routes/analytics.py
# Adjust query time ranges for better performance
week_ago = now - timedelta(days=7)  # Can be adjusted
month_ago = now - timedelta(days=30)  # Can be adjusted

# Limit query results to prevent memory issues
POPULAR_QUERIES_LIMIT = 10  # Top queries to display
RECENT_QUERIES_LIMIT = 5    # Recent activity items
```

### Frontend Refresh Intervals
```typescript
// src/web-ui/src/pages/Analytics.tsx
const REALTIME_REFRESH_INTERVAL = 30000  // 30 seconds
const ANALYTICS_REFRESH_INTERVAL = 300000 // 5 minutes (full refresh)
```

## Security Considerations

### Data Privacy
- Search queries may contain sensitive information
- Consider data retention policies for search_history table
- Implement query sanitization if displaying in public dashboards

### Access Control
- Analytics page requires authentication (if implemented)
- API endpoints follow same security model as other routes
- Consider role-based access for sensitive metrics

### Performance Impact
- Analytics recording is designed to be non-blocking
- Database queries are optimized with proper indexes
- Frontend caching reduces API load

## Future Enhancements

### Planned Features
- **Export Functionality**: Download analytics as CSV/PDF
- **Advanced Filtering**: Filter analytics by date ranges, users, sources
- **Alert System**: Notifications for performance thresholds
- **Search Suggestions**: AI-powered query recommendations based on analytics
- **A/B Testing**: Compare different search algorithms
- **User Analytics**: Track individual user search patterns (with privacy controls)

### Technical Improvements
- **Real-time WebSocket Updates**: Live metrics without polling
- **Advanced Caching**: Redis-based analytics caching
- **Data Aggregation**: Pre-computed daily/weekly statistics
- **Mobile Optimization**: Responsive charts for mobile devices
- **Custom Dashboards**: User-configurable analytics views

## Related Documentation
- [Search System Architecture](SEARCH_ARCHITECTURE.md)
- [Performance Monitoring](PERFORMANCE_METRICS_API.md)
- [Database Schema](../src/api/database/schema.sql)
- [Frontend Components](../src/web-ui/src/pages/Analytics.tsx)