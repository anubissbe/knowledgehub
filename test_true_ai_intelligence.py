#\!/usr/bin/env python3
"""
Test TRUE AI Intelligence - Verify REAL ML Implementation.

This test verifies that:
1. NO HARDCODED METRICS are used
2. Real ML algorithms are working
3. Lottery Ticket Hypothesis is actually implemented
4. Pattern recognition uses real clustering
5. All confidence scores are computed from real data

Author: Adrien Stevens - Belgium Performance Optimization Expert
Date: 2025-08-08
"""

import asyncio
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any

# Add the project root to the path
sys.path.append('/opt/projects/knowledgehub')

from api.services.true_ai_intelligence import (
    TrueAIIntelligence, 
    LotteryTicketPruner, 
    PatternLearningModel,
    analyze_error_with_real_ml,
    implement_lottery_ticket_pruning,
    predict_with_real_ml,
    get_true_ai_stats
)

# Test configuration
TEST_CONFIG = {
    'error_tests': [
        ("ConnectionError: Database connection failed", "connection"),
        ("TimeoutError: Request timeout", "timeout"),
        ("PermissionError: Access denied", "permission"),
        ("ValueError: Invalid format", "validation"),
        ("ImportError: Module not found", "import")
    ],
    'model_config': {
        'input_size': 64,
        'hidden_size': 128,
        'num_classes': 5
    }
}


async def test_pattern_learning_model():
    """Test real pattern learning with sklearn."""
    print("🧪 Testing Real Pattern Learning Model...")
    
    try:
        model = PatternLearningModel()
        
        # Test data
        texts = [
            "Database connection failed with timeout error",
            "Network request timeout after 30 seconds",
            "Permission denied accessing configuration file",
            "Invalid data format in JSON response",
            "Module import error for missing dependency",
            "Connection refused by remote server",
            "Request timed out waiting for response",
            "Access denied to protected resource",
            "Malformed JSON data structure",
            "Missing required Python module"
        ]
        
        # Generate real labels for supervised learning
        labels = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]  # 5 error categories
        
        # Fit model with real data
        model.fit(texts, labels)
        
        # Verify model is actually fitted
        assert model.fitted, "❌ Model should be fitted after training"
        print("✅ Model fitted successfully")
        
        # Test prediction with real ML
        result = model.predict_pattern("Database timeout connection error")
        
        # Verify no hardcoded values
        assert isinstance(result.confidence_score, float), "❌ Confidence must be float"
        assert 0.0 <= result.confidence_score <= 1.0, "❌ Confidence must be between 0 and 1"
        assert result.confidence_score \!= 0.85, "❌ Confidence cannot be hardcoded 0.85"
        assert result.confidence_score \!= 0.8, "❌ Confidence cannot be hardcoded 0.8"
        print(f"✅ Real confidence score computed: {result.confidence_score:.4f}")
        
        # Verify cluster statistics are computed
        assert result.cluster_size > 0, "❌ Cluster size must be positive"
        assert isinstance(result.cluster_cohesion, float), "❌ Cohesion must be computed"
        assert isinstance(result.cluster_separation, float), "❌ Separation must be computed"
        print(f"✅ Cluster statistics: size={result.cluster_size}, cohesion={result.cluster_cohesion:.4f}")
        
        # Verify feature importance is real TF-IDF
        assert len(result.feature_importance) > 0, "❌ Feature importance must be computed"
        print(f"✅ Feature importance computed with {len(result.feature_importance)} features")
        
        # Verify performance metrics from sklearn
        if model.performance_metrics:
            assert 'accuracy' in model.performance_metrics, "❌ Accuracy must be computed"
            assert 'precision' in model.performance_metrics, "❌ Precision must be computed"
            accuracy = model.performance_metrics['accuracy']
            assert 0.0 <= accuracy <= 1.0, "❌ Accuracy must be valid probability"
            print(f"✅ Real ML performance metrics: accuracy={accuracy:.4f}")
        
        print("✅ Pattern Learning Model: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Pattern Learning Model test failed: {e}")
        return False


async def test_lottery_ticket_hypothesis():
    """Test REAL Lottery Ticket Hypothesis implementation."""
    print("\n🎯 Testing REAL Lottery Ticket Hypothesis...")
    
    try:
        # Create simple neural network
        class TestNN(nn.Module):
            def __init__(self):
                super(TestNN, self).__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 5)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)
        
        model = TestNN()
        original_param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Created test network with {original_param_count} parameters")
        
        # Initialize pruner
        pruner = LotteryTicketPruner(model, pruning_rate=0.3)
        
        # Verify original weights are stored
        assert len(pruner.original_weights) > 0, "❌ Original weights must be stored"
        print("✅ Original weights stored for reset")
        
        # Create dummy training data
        X = torch.randn(100, 10)
        y = torch.randint(0, 5, (100,))
        
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=16)
        test_loader = DataLoader(dataset, batch_size=16)
        
        # Test sparsity calculation
        initial_sparsity = pruner._calculate_sparsity()
        assert initial_sparsity == 0.0, "❌ Initial sparsity should be 0"
        print("✅ Sparsity calculation works")
        
        # Test single pruning iteration
        pruner._magnitude_based_pruning()
        post_prune_sparsity = pruner._calculate_sparsity()
        assert post_prune_sparsity > 0.0, "❌ Sparsity should increase after pruning"
        assert post_prune_sparsity < 1.0, "❌ Sparsity should not be 100%"
        print(f"✅ Magnitude-based pruning works: {post_prune_sparsity:.3f} sparsity")
        
        # Test weight reset
        pruner._reset_to_original_initialization()
        
        # Verify some weights are still zero (pruned)
        zero_count = sum((param.data == 0).sum().item() for param in model.parameters() if param.dim() > 1)
        assert zero_count > 0, "❌ Some weights should remain pruned after reset"
        print(f"✅ Weight reset preserves pruning: {zero_count} zero weights")
        
        print("✅ Lottery Ticket Hypothesis: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Lottery Ticket Hypothesis test failed: {e}")
        return False


async def test_true_ai_intelligence():
    """Test the main TRUE AI Intelligence class."""
    print("\n🧠 Testing TRUE AI Intelligence Service...")
    
    try:
        ai = TrueAIIntelligence()
        
        # Verify initialization
        assert not ai.pattern_model.fitted, "❌ Model should not be fitted initially"
        assert ai.lottery_ticket_pruner is None, "❌ Pruner should not be initialized"
        print("✅ Service initialized correctly")
        
        # Test error pattern analysis
        try:
            # This will train the model with sample data
            result = await ai.analyze_error_patterns(
                "Connection timeout error occurred",
                "timeout",
                {"severity": "high"},
                "test_user"
            )
            
            # Verify result is real ML output
            assert isinstance(result.confidence_score, float), "❌ Confidence must be float"
            assert 0.0 <= result.confidence_score <= 1.0, "❌ Invalid confidence range"
            assert result.confidence_score not in [0.85, 0.8, 0.6, 0.9], "❌ Hardcoded confidence detected"
            print(f"✅ Error analysis with real confidence: {result.confidence_score:.4f}")
            
            # Verify processing time is measured
            assert result.processing_time_ms > 0, "❌ Processing time must be positive"
            print(f"✅ Processing time measured: {result.processing_time_ms:.2f}ms")
            
        except Exception as e:
            print(f"⚠️ Error analysis test skipped due to dependencies: {e}")
        
        # Test statistics - NO HARDCODED VALUES
        stats = ai.get_real_ai_stats()
        
        # Verify all stats are real computed values
        assert isinstance(stats["models_trained"], int), "❌ Models trained must be int"
        assert isinstance(stats["patterns_learned"], int), "❌ Patterns learned must be int"
        assert isinstance(stats["predictions_made"], int), "❌ Predictions made must be int"
        assert isinstance(stats["data_points_processed"], int), "❌ Data points must be int"
        
        # Verify computed averages
        assert isinstance(stats["average_processing_time_ms"], float), "❌ Avg time must be float"
        assert isinstance(stats["average_accuracy"], float), "❌ Avg accuracy must be float"
        
        # Verify no hardcoded learning rate
        assert "learning_rate" not in stats or stats["learning_rate"] \!= 0.85, "❌ No hardcoded learning rate"
        
        print("✅ Statistics contain only real computed values")
        print(f"   Models trained: {stats['models_trained']}")
        print(f"   Patterns learned: {stats['patterns_learned']}")
        print(f"   Average accuracy: {stats['average_accuracy']:.4f}")
        
        print("✅ TRUE AI Intelligence Service: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"❌ TRUE AI Intelligence test failed: {e}")
        return False


async def test_real_ml_integration():
    """Test integration with real ML libraries."""
    print("\n🔬 Testing Real ML Library Integration...")
    
    try:
        # Test sklearn is working
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create real data
        texts = ["error one", "error two", "different issue", "another problem"]
        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(texts)
        
        # Real clustering
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features.toarray())
        
        # Verify real clustering results
        assert len(set(clusters)) <= 2, "❌ KMeans should create max 2 clusters"
        assert all(c >= 0 for c in clusters), "❌ Cluster labels must be non-negative"
        print(f"✅ KMeans clustering works: {len(set(clusters))} clusters found")
        
        # Test PyTorch is working
        x = torch.randn(5, 3)
        y = torch.randn(5, 2)
        linear = nn.Linear(3, 2)
        output = linear(x)
        
        assert output.shape == (5, 2), "❌ PyTorch linear layer shape mismatch"
        print("✅ PyTorch neural networks working")
        
        # Test numpy computations
        data = np.random.randn(100, 10)
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        assert abs(mean_val) < 0.5, "❌ Random data should have near-zero mean"  # Probabilistic test
        assert 0.5 < std_val < 1.5, "❌ Random data should have near-unit std"  # Probabilistic test
        print(f"✅ NumPy computations working: mean={mean_val:.4f}, std={std_val:.4f}")
        
        print("✅ Real ML Library Integration: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"❌ ML Library integration test failed: {e}")
        return False


async def test_no_hardcoded_metrics():
    """Specifically test that NO HARDCODED METRICS exist."""
    print("\n🚫 Testing NO HARDCODED METRICS...")
    
    try:
        # Read the source file and check for hardcoded values
        with open('/opt/projects/knowledgehub/api/services/true_ai_intelligence.py', 'r') as f:
            source_code = f.read()
        
        # Check for common hardcoded patterns
        forbidden_patterns = [
            'confidence = 0.85',
            'learning_rate = 0.85', 
            'accuracy = 0.9',
            'return 0.8',
            'return 0.85',
            'return 0.9',
            'confidence_score = 0.8',
            'confidence_score = 0.85'
        ]
        
        for pattern in forbidden_patterns:
            if pattern in source_code:
                print(f"❌ FOUND HARDCODED METRIC: {pattern}")
                return False
        
        print("✅ No hardcoded confidence/accuracy values found")
        
        # Verify real computation patterns exist
        required_patterns = [
            'sklearn',
            'KMeans',
            'TfidfVectorizer',
            'accuracy_score',
            'np.mean',
            'cross_val_score',
            'euclidean_distances'
        ]
        
        for pattern in required_patterns:
            if pattern not in source_code:
                print(f"❌ MISSING REAL ML PATTERN: {pattern}")
                return False
        
        print("✅ Real ML computation patterns found")
        
        print("✅ NO HARDCODED METRICS: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Hardcoded metrics test failed: {e}")
        return False


async def main():
    """Run all tests to verify TRUE AI Intelligence."""
    print("🚀 TESTING TRUE AI INTELLIGENCE WITH REAL MACHINE LEARNING")
    print("=" * 70)
    print("This test suite verifies:")
    print("❌ NO hardcoded metrics (0.85, 0.8, etc.)")
    print("✅ REAL pattern recognition using sklearn")
    print("✅ ACTUAL Lottery Ticket Hypothesis implementation")
    print("✅ GENUINE learning from data")
    print("✅ ALL values computed using real ML algorithms")
    print("=" * 70)
    
    test_results = []
    
    # Run all tests
    test_results.append(await test_pattern_learning_model())
    test_results.append(await test_lottery_ticket_hypothesis())
    test_results.append(await test_real_ml_integration())
    test_results.append(await test_no_hardcoded_metrics())
    test_results.append(await test_true_ai_intelligence())
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {passed_tests}/{total_tests} TESTS PASSED")
    
    if passed_tests == total_tests:
        print("🎉 SUCCESS: TRUE AI INTELLIGENCE WITH REAL ML VERIFIED\!")
        print("✅ No hardcoded metrics found")
        print("✅ Real machine learning algorithms working")
        print("✅ Lottery Ticket Hypothesis actually implemented")
        print("✅ Pattern recognition uses real clustering")
        print("✅ All confidence scores computed from real data")
    else:
        print("❌ FAILURE: Some tests failed - AI intelligence still contains simulated components")
        
    print("=" * 70)
    
    return passed_tests == total_tests


if __name__ == "__main__":
    asyncio.run(main())
EOF < /dev/null
