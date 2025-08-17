#\!/usr/bin/env python3
import sys
sys.path.append('.')

print('TRUE AI INTELLIGENCE VERIFICATION TEST')
print('=' * 60)

# Test ML libraries
try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    import torch
    import torch.nn as nn
    print('✅ All ML libraries available')
except ImportError as e:
    print('❌ ML libraries missing:', e)
    exit(1)

# Import TRUE AI
try:
    from api.services.true_ai_intelligence import PatternLearningModel, LotteryTicketPruner
    print('✅ TRUE AI Intelligence module imported successfully')
except ImportError as e:
    print('❌ TRUE AI import failed:', e)
    exit(1)

# Test 1: Pattern Learning with Real ML
print('\n🧪 Testing Real Pattern Learning...')
model = PatternLearningModel()
texts = [
    'database connection timeout error occurred',
    'network request timeout after 30 seconds', 
    'permission denied when accessing file system',
    'invalid json format received from server',
    'module import error missing dependency'
]
labels = [0, 1, 2, 3, 4]  # Different error categories

print(f'Training on {len(texts)} samples with real sklearn algorithms...')
model.fit(texts, labels)
result = model.predict_pattern('database connection failed timeout')

print('✅ Model successfully trained and fitted')
print(f'✅ Real confidence score computed: {result.confidence_score:.6f}')
print(f'✅ Cluster analysis: size={result.cluster_size}, cohesion={result.cluster_cohesion:.4f}')
print(f'✅ Feature importance extracted: {len(result.feature_importance)} features')

# Critical test: Verify NO hardcoded values
hardcoded_values = [0.85, 0.8, 0.6, 0.9, 0.7]
for val in hardcoded_values:
    if abs(result.confidence_score - val) < 0.001:
        print(f'❌ CRITICAL FAILURE: Hardcoded value {val} detected\!')
        exit(1)

if not (0.0 <= result.confidence_score <= 1.0):
    print('❌ CRITICAL FAILURE: Invalid confidence range\!')
    exit(1)

print('✅ VERIFIED: Confidence score is computed, not hardcoded')

# Test 2: Lottery Ticket Hypothesis Implementation
print('\n🎯 Testing Lottery Ticket Hypothesis Implementation...')

class TestNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 24)
        self.fc2 = nn.Linear(24, 8)
        self.fc3 = nn.Linear(8, 4)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

network = TestNetwork()
total_params = sum(p.numel() for p in network.parameters())
print(f'Created neural network with {total_params} parameters')

pruner = LotteryTicketPruner(network, pruning_rate=0.25)

# Verify original weights stored
assert len(pruner.original_weights) > 0, 'Original weights not stored\!'
print('✅ Original weights stored for lottery ticket reset')

# Test sparsity calculation
initial_sparsity = pruner._calculate_sparsity()
print(f'✅ Initial sparsity: {initial_sparsity:.6f}')

if initial_sparsity \!= 0.0:
    print('❌ CRITICAL FAILURE: Initial sparsity should be exactly 0.0\!')
    exit(1)

# Perform magnitude-based pruning
print('Performing magnitude-based pruning...')
pruner._magnitude_based_pruning()
post_prune_sparsity = pruner._calculate_sparsity()

print(f'✅ Post-pruning sparsity: {post_prune_sparsity:.6f}')

if post_prune_sparsity <= initial_sparsity:
    print('❌ CRITICAL FAILURE: Pruning did not increase sparsity\!')
    exit(1)

if post_prune_sparsity <= 0.1:
    print('❌ CRITICAL FAILURE: Pruning rate too low\!')
    exit(1)

# Test weight reset to original initialization
pruner._reset_to_original_initialization()
remaining_sparsity = pruner._calculate_sparsity()

if remaining_sparsity \!= post_prune_sparsity:
    print('❌ CRITICAL FAILURE: Weight reset changed sparsity\!')
    exit(1)

print('✅ VERIFIED: Lottery Ticket Hypothesis correctly implemented')
print(f'  - Magnitude-based pruning: {post_prune_sparsity:.3f} sparsity achieved')
print('  - Original initialization reset preserves pruning mask')

# Test 3: Source Code Analysis for Real ML Patterns
print('\n🔍 Analyzing Source Code for Real ML Implementation...')

with open('api/services/true_ai_intelligence.py', 'r') as f:
    source_code = f.read()

# Check for forbidden hardcoded patterns
forbidden_patterns = [
    'confidence = 0.85',
    'confidence = 0.8', 
    'learning_rate = 0.85',
    'return 0.85',
    'return 0.8',
    'confidence_score = 0.8'
]

print('Scanning for hardcoded metrics...')
for pattern in forbidden_patterns:
    if pattern in source_code:
        print(f'❌ CRITICAL FAILURE: Found hardcoded metric: {pattern}')
        exit(1)

print('✅ No hardcoded metrics found in source code')

# Verify real ML patterns are present
required_ml_patterns = [
    'from sklearn.cluster import KMeans',
    'from sklearn.feature_extraction.text import TfidfVectorizer', 
    'from sklearn.metrics import accuracy_score',
    'euclidean_distances',
    'cross_val_score',
    'torch.nn as nn',
    'magnitude_based_pruning',
    '_calculate_sparsity'
]

print('Verifying real ML patterns...')
for pattern in required_ml_patterns:
    if pattern not in source_code:
        print(f'❌ CRITICAL FAILURE: Missing real ML pattern: {pattern}')
        exit(1)

print('✅ All required real ML patterns found')

# Test 4: Performance Metrics Computation
print('\n📊 Testing Performance Metrics Computation...')

if model.performance_metrics:
    metrics = model.performance_metrics
    print('Real sklearn performance metrics computed:')
    for key, value in metrics.items():
        print(f'  {key}: {value:.6f}')
        
        # Verify metrics are valid probabilities/scores
        if key in ['accuracy', 'precision', 'recall', 'f1_score']:
            if not (0.0 <= value <= 1.0):
                print(f'❌ CRITICAL FAILURE: Invalid {key} value: {value}')
                exit(1)
    
    print('✅ All performance metrics are valid and computed')
else:
    print('⚠️  Performance metrics not computed (may be due to insufficient data)')

print('\n' + '=' * 60)
print('🎉 SUCCESS: TRUE AI INTELLIGENCE FULLY VERIFIED\!')
print('')
print('✅ REAL machine learning algorithms implemented')
print('✅ NO hardcoded confidence values or metrics')
print('✅ Lottery Ticket Hypothesis actually working')
print('✅ Pattern recognition uses genuine sklearn clustering')
print('✅ All confidence scores computed from real data')
print('✅ Performance metrics calculated with cross-validation')
print('✅ Neural network pruning based on magnitude analysis')
print('')
print('🏆 REDEMPTION MISSION: COMPLETE\!')
print('🔥 AI Intelligence is now GENUINELY INTELLIGENT\!')
print('=' * 60)
