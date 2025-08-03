#!/usr/bin/env python3
"""
Test TimescaleDB integration to ensure all features are working properly.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import random

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.services.time_series_analytics import TimeSeriesAnalyticsService, MetricType, AggregationWindow


async def test_timescale_integration():
    """Test all TimescaleDB functionality."""
    print("🔧 Testing TimescaleDB Integration...")
    
    # Initialize service
    service = TimeSeriesAnalyticsService()
    
    try:
        # Initialize connection
        await service.initialize()
        print("✅ TimescaleDB connection initialized")
        
        # Test recording metrics
        print("\n📊 Testing metric recording...")
        for i in range(10):
            await service.record_metric(
                MetricType.KNOWLEDGE_CREATION,
                random.uniform(0.5, 1.0),
                tags={'test': 'true', 'batch': str(i // 5)},
                metadata={'description': f'Test metric {i}'}
            )
        print("✅ Recorded 10 test metrics")
        
        # Test recording knowledge evolution
        print("\n🧠 Testing knowledge evolution recording...")
        await service.record_knowledge_evolution(
            'test_entity',
            'entity_1',
            'create',
            old_value={},
            new_value={'status': 'created'},
            confidence=0.95
        )
        print("✅ Recorded knowledge evolution event")
        
        # Test recording pattern trends
        print("\n🔍 Testing pattern trend recording...")
        await service.record_pattern_trend(
            'test_pattern',
            'optimization_sequence',
            occurrence_count=3,
            effectiveness_score=0.85,
            context={'project': 'knowledgehub'}
        )
        print("✅ Recorded pattern trend")
        
        # Test recording performance metrics
        print("\n⚡ Testing performance recording...")
        await service.record_performance(
            '/api/test',
            response_time=0.123,
            memory_usage=50.0,
            cpu_usage=25.0,
            request_count=1
        )
        print("✅ Recorded performance metrics")
        
        # Test metric trends
        print("\n📈 Testing metric trend analysis...")
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        trends = await service.get_metric_trends(
            MetricType.KNOWLEDGE_CREATION,
            start_time,
            end_time,
            AggregationWindow.MINUTE
        )
        print(f"✅ Retrieved {len(trends)} trend data points")
        
        # Test trend analysis
        print("\n🔍 Testing trend analysis...")
        analysis = await service.analyze_trends(
            MetricType.KNOWLEDGE_CREATION,
            start_time,
            end_time,
            AggregationWindow.MINUTE
        )
        print(f"✅ Trend analysis: {analysis.trend_direction} trend with {analysis.trend_strength:.2f} strength")
        
        # Test knowledge evolution timeline
        print("\n📜 Testing knowledge evolution timeline...")
        evolution = await service.get_knowledge_evolution_timeline(
            entity_type='test_entity',
            limit=10
        )
        print(f"✅ Retrieved {len(evolution)} evolution events")
        
        # Test pattern effectiveness trends
        print("\n📊 Testing pattern effectiveness trends...")
        pattern_trends = await service.get_pattern_effectiveness_trends(
            pattern_type='test_pattern',
            start_time=start_time,
            end_time=end_time
        )
        print(f"✅ Retrieved pattern trends for {len(pattern_trends)} patterns")
        
        # Test dashboard data generation
        print("\n📋 Testing dashboard data generation...")
        dashboard_data = await service.generate_analytics_dashboard_data()
        print(f"✅ Generated dashboard data with {len(dashboard_data['overview'])} overview metrics")
        
        # Test retention policy status
        print("\n⏰ Testing retention policy status...")
        retention_status = await service.get_retention_policy_status()
        print(f"✅ Found {retention_status['total_policies']} retention policies")
        
        # Test hypertable information
        print("\n🏗️ Testing hypertable information...")
        hypertable_info = await service.get_hypertable_info()
        print(f"✅ Found {hypertable_info['total_hypertables']} hypertables")
        
        if hypertable_info['hypertables']:
            for ht in hypertable_info['hypertables']:
                size_mb = ht['total_size_bytes'] / 1024 / 1024 if ht['total_size_bytes'] else 0
                print(f"   - {ht['name']}: {ht['chunks']} chunks, {size_mb:.2f} MB")
        
        print("\n🎉 All TimescaleDB tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ TimescaleDB test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await service.cleanup()


async def main():
    """Main test function."""
    success = await test_timescale_integration()
    
    if success:
        print("\n✅ TimescaleDB integration is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ TimescaleDB integration has issues!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())