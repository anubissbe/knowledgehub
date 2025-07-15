-- TimescaleDB Analytics Tables
-- Run this on the TimescaleDB instance (port 5434)

-- Create TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Performance metrics time-series
CREATE TABLE IF NOT EXISTS performance_timeseries (
    time TIMESTAMPTZ NOT NULL,
    user_id UUID,
    metric_name VARCHAR(100) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    tags JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- Convert to hypertable
SELECT create_hypertable('performance_timeseries', 'time');

-- API request metrics
CREATE TABLE IF NOT EXISTS api_metrics (
    time TIMESTAMPTZ NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER,
    response_time_ms INTEGER,
    user_id UUID,
    metadata JSONB DEFAULT '{}'
);

SELECT create_hypertable('api_metrics', 'time');

-- Learning effectiveness metrics
CREATE TABLE IF NOT EXISTS learning_metrics (
    time TIMESTAMPTZ NOT NULL,
    user_id UUID,
    metric_type VARCHAR(50),
    score DOUBLE PRECISION,
    context JSONB DEFAULT '{}'
);

SELECT create_hypertable('learning_metrics', 'time');

-- System health metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    time TIMESTAMPTZ NOT NULL,
    service_name VARCHAR(100),
    metric_name VARCHAR(100),
    value DOUBLE PRECISION,
    status VARCHAR(20)
);

SELECT create_hypertable('system_metrics', 'time');

-- Create continuous aggregates for common queries
CREATE MATERIALIZED VIEW hourly_performance
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS hour,
    user_id,
    metric_name,
    avg(value) as avg_value,
    max(value) as max_value,
    min(value) as min_value,
    count(*) as count
FROM performance_timeseries
GROUP BY hour, user_id, metric_name;

CREATE MATERIALIZED VIEW daily_api_stats
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', time) AS day,
    endpoint,
    method,
    count(*) as request_count,
    avg(response_time_ms) as avg_response_time,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY response_time_ms) as p99_response_time
FROM api_metrics
GROUP BY day, endpoint, method;

-- Add retention policy (keep 90 days of raw data)
SELECT add_retention_policy('performance_timeseries', INTERVAL '90 days');
SELECT add_retention_policy('api_metrics', INTERVAL '90 days');
SELECT add_retention_policy('learning_metrics', INTERVAL '90 days');
SELECT add_retention_policy('system_metrics', INTERVAL '30 days');