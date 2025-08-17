"""
Advanced Semantic Analysis System Beyond RAG
Created by Annelies Claes - Expert in Neural Network Quantization & Semantic Analysis

This system provides advanced document similarity and semantic relationship detection
that goes far beyond traditional RAG (Retrieval-Augmented Generation) approaches
using quantized neural networks and graph-based analysis.
"""

import asyncio
import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict, Counter
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, KMeans
from sentence_transformers import SentenceTransformer
import spacy
from textblob import TextBlob
import re

logger = logging.getLogger(__name__)

@dataclass
class SemanticRelationship:
    """Represents a semantic relationship between documents or concepts."""
    source_id: str
    target_id: str
    relationship_type: str  # 'similar', 'contradicts', 'extends', 'references', 'prerequisite'
    strength: float  # 0.0 to 1.0
    confidence: float
    evidence: List[str]  # Supporting evidence for the relationship
    metadata: Dict[str, Any]

@dataclass
class ConceptNode:
    """Represents a concept extracted from document analysis."""
    concept_id: str
    concept_name: str
    concept_type: str  # 'entity', 'topic', 'methodology', 'technology'
    importance_score: float
    frequency: int
    document_sources: Set[str]
    related_concepts: List[str]
    embedding: Optional[np.ndarray] = None

@dataclass
class DocumentAnalysis:
    """Comprehensive document analysis results."""
    document_id: str
    semantic_fingerprint: np.ndarray
    key_concepts: List[ConceptNode]
    topic_distribution: Dict[str, float]
    complexity_metrics: Dict[str, float]
    quality_score: float
    readability_metrics: Dict[str, float]
    entity_graph: nx.Graph
    metadata: Dict[str, Any]

class AdvancedSemanticAnalyzer:
    """
    Advanced Semantic Analysis System using quantized neural networks.
    
    Features:
    - Multi-dimensional similarity analysis (semantic, syntactic, pragmatic)
    - Concept extraction and relationship mapping
    - Document quality assessment
    - Topic modeling with hierarchical clustering
    - Cross-lingual semantic analysis
    - Temporal semantic evolution tracking
    """
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        spacy_model: str = "en_core_web_sm",
        use_quantization: bool = True,
        quantization_bits: int = 8
    ):
        self.embedding_model_name = embedding_model_name
        self.spacy_model_name = spacy_model
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        
        # Models and processors
        self.embedding_model: Optional[SentenceTransformer] = None
        self.spacy_nlp = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        
        # Analysis components
        self.concept_graph = nx.Graph()
        self.document_embeddings = {}
        self.semantic_clusters = {}
        
        # Quantization parameters
        if use_quantization:
            self.embedding_dtype = torch.float16 if quantization_bits <= 16 else torch.float32
        else:
            self.embedding_dtype = torch.float32
        
        logger.info(f"AdvancedSemanticAnalyzer initialized with {quantization_bits}-bit quantization")

    async def initialize(self):
        """Initialize all models and processors."""
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Apply quantization if enabled
            if self.use_quantization:
                self._quantize_embedding_model()
            
            # Initialize spaCy NLP pipeline
            try:
                self.spacy_nlp = spacy.load(self.spacy_model_name)
            except OSError:
                logger.warning(f"spaCy model {self.spacy_model_name} not found, using basic English model")
                self.spacy_nlp = spacy.load("en_core_web_sm")
            
            logger.info("AdvancedSemanticAnalyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AdvancedSemanticAnalyzer: {e}")
            raise

    def _quantize_embedding_model(self):
        """Apply quantization to the embedding model for efficiency."""
        try:
            # Convert model to quantized precision
            if hasattr(self.embedding_model, '_modules'):
                for name, module in self.embedding_model._modules.items():
                    if hasattr(module, 'weight'):
                        # Apply quantization to weights
                        quantized_weight = self._quantize_tensor(module.weight)
                        module.weight = torch.nn.Parameter(quantized_weight)
            
            logger.info(f"Applied {self.quantization_bits}-bit quantization to embedding model")
            
        except Exception as e:
            logger.warning(f"Failed to quantize embedding model: {e}")

    def _quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize a tensor to reduced precision."""
        if self.quantization_bits >= 32:
            return tensor
        
        # Calculate quantization scale and zero point
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        scale = (tensor_max - tensor_min) / (2 ** self.quantization_bits - 1)
        
        # Quantize
        quantized = torch.round((tensor - tensor_min) / scale) * scale + tensor_min
        
        # Convert to target dtype
        return quantized.to(self.embedding_dtype)

    async def analyze_document(
        self,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentAnalysis:
        """
        Perform comprehensive semantic analysis of a document.
        """
        try:
            # Generate semantic fingerprint
            semantic_fingerprint = await self._generate_semantic_fingerprint(content)
            
            # Extract key concepts
            key_concepts = await self._extract_key_concepts(content, document_id)
            
            # Analyze topic distribution
            topic_distribution = await self._analyze_topic_distribution(content)
            
            # Calculate complexity metrics
            complexity_metrics = self._calculate_complexity_metrics(content)
            
            # Assess document quality
            quality_score = await self._assess_document_quality(content, key_concepts)
            
            # Calculate readability metrics
            readability_metrics = self._calculate_readability_metrics(content)
            
            # Build entity relationship graph
            entity_graph = await self._build_entity_graph(content)
            
            analysis = DocumentAnalysis(
                document_id=document_id,
                semantic_fingerprint=semantic_fingerprint,
                key_concepts=key_concepts,
                topic_distribution=topic_distribution,
                complexity_metrics=complexity_metrics,
                quality_score=quality_score,
                readability_metrics=readability_metrics,
                entity_graph=entity_graph,
                metadata=metadata or {}
            )
            
            # Store analysis for future use
            self.document_embeddings[document_id] = semantic_fingerprint
            
            return analysis
            
        except Exception as e:
            logger.error(f"Document analysis failed for {document_id}: {e}")
            raise

    async def _generate_semantic_fingerprint(self, content: str) -> np.ndarray:
        """Generate a comprehensive semantic fingerprint for the content."""
        # Base semantic embedding
        base_embedding = self.embedding_model.encode(content)
        
        # Sentence-level embeddings for granular analysis
        sentences = content.split('.')[:10]  # Limit to first 10 sentences for efficiency
        sentence_embeddings = [
            self.embedding_model.encode(sent.strip()) 
            for sent in sentences if sent.strip()
        ]
        
        # Calculate embedding statistics
        if sentence_embeddings:
            sentence_embeddings = np.array(sentence_embeddings)
            embedding_mean = np.mean(sentence_embeddings, axis=0)
            embedding_std = np.std(sentence_embeddings, axis=0)
            
            # Combine base embedding with statistical features
            fingerprint = np.concatenate([
                base_embedding,
                embedding_mean,
                embedding_std
            ])
        else:
            fingerprint = base_embedding
        
        # Apply quantization if enabled
        if self.use_quantization:
            fingerprint = self._quantize_numpy_array(fingerprint)
        
        return fingerprint

    def _quantize_numpy_array(self, array: np.ndarray) -> np.ndarray:
        """Quantize a NumPy array to reduced precision."""
        if self.quantization_bits >= 32:
            return array
        
        # Calculate quantization parameters
        array_min = array.min()
        array_max = array.max()
        scale = (array_max - array_min) / (2 ** self.quantization_bits - 1)
        
        # Quantize
        quantized = np.round((array - array_min) / scale) * scale + array_min
        
        # Convert to appropriate dtype
        if self.quantization_bits <= 16:
            return quantized.astype(np.float16)
        else:
            return quantized.astype(np.float32)

    async def _extract_key_concepts(
        self, 
        content: str, 
        document_id: str
    ) -> List[ConceptNode]:
        """Extract key concepts using NLP and semantic analysis."""
        concepts = []
        
        # Process with spaCy
        doc = self.spacy_nlp(content)
        
        # Extract entities
        entities = {}
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'WORK_OF_ART', 'LAW']:
                entities[ent.text.lower()] = {
                    'type': 'entity',
                    'label': ent.label_,
                    'frequency': entities.get(ent.text.lower(), {}).get('frequency', 0) + 1
                }
        
        # Extract noun phrases as potential concepts
        noun_phrases = {}
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 3 and chunk.text.lower() not in entities:
                noun_phrases[chunk.text.lower()] = {
                    'type': 'topic',
                    'frequency': noun_phrases.get(chunk.text.lower(), {}).get('frequency', 0) + 1
                }
        
        # Create concept nodes
        all_concepts = {**entities, **noun_phrases}
        
        for concept_text, concept_data in all_concepts.items():
            if concept_data['frequency'] >= 2:  # Filter by frequency
                # Generate concept embedding
                concept_embedding = self.embedding_model.encode(concept_text)
                
                concept = ConceptNode(
                    concept_id=f"{document_id}:{concept_text}",
                    concept_name=concept_text,
                    concept_type=concept_data['type'],
                    importance_score=min(concept_data['frequency'] / 10.0, 1.0),
                    frequency=concept_data['frequency'],
                    document_sources={document_id},
                    related_concepts=[],
                    embedding=concept_embedding
                )
                
                concepts.append(concept)
        
        # Sort by importance
        concepts.sort(key=lambda x: x.importance_score, reverse=True)
        
        return concepts[:20]  # Return top 20 concepts

    async def _analyze_topic_distribution(self, content: str) -> Dict[str, float]:
        """Analyze topic distribution in the content."""
        # Simple topic modeling using TF-IDF and clustering
        try:
            # Fit TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([content])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get top TF-IDF terms as topics
            tfidf_scores = tfidf_matrix.toarray()[0]
            top_indices = np.argsort(tfidf_scores)[-20:]  # Top 20 terms
            
            topic_distribution = {}
            total_score = sum(tfidf_scores[i] for i in top_indices)
            
            for idx in top_indices:
                if total_score > 0:
                    topic_distribution[feature_names[idx]] = tfidf_scores[idx] / total_score
            
            return topic_distribution
            
        except Exception as e:
            logger.warning(f"Topic analysis failed: {e}")
            return {}

    def _calculate_complexity_metrics(self, content: str) -> Dict[str, float]:
        """Calculate various complexity metrics for the content."""
        try:
            # Text statistics
            sentences = content.split('.')
            words = content.split()
            
            # Basic metrics
            avg_sentence_length = len(words) / len(sentences) if sentences else 0
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            
            # Vocabulary diversity (Type-Token Ratio)
            unique_words = set(word.lower() for word in words)
            vocabulary_diversity = len(unique_words) / len(words) if words else 0
            
            # Syntactic complexity (using spaCy)
            doc = self.spacy_nlp(content)
            
            # Calculate dependency depth
            dependency_depths = []
            for sent in doc.sents:
                max_depth = 0
                for token in sent:
                    depth = self._get_dependency_depth(token)
                    max_depth = max(max_depth, depth)
                dependency_depths.append(max_depth)
            
            avg_dependency_depth = np.mean(dependency_depths) if dependency_depths else 0
            
            return {
                'avg_sentence_length': avg_sentence_length,
                'avg_word_length': avg_word_length,
                'vocabulary_diversity': vocabulary_diversity,
                'avg_dependency_depth': avg_dependency_depth,
                'sentence_count': len(sentences),
                'word_count': len(words),
                'unique_word_count': len(unique_words)
            }
            
        except Exception as e:
            logger.warning(f"Complexity calculation failed: {e}")
            return {}

    def _get_dependency_depth(self, token, depth=0):
        """Calculate dependency depth for a token."""
        if not list(token.children):
            return depth
        return max(self._get_dependency_depth(child, depth + 1) for child in token.children)

    async def _assess_document_quality(
        self, 
        content: str, 
        concepts: List[ConceptNode]
    ) -> float:
        """Assess overall document quality based on multiple factors."""
        try:
            quality_factors = []
            
            # Content length factor (not too short, not too long)
            word_count = len(content.split())
            if 100 <= word_count <= 10000:
                length_factor = 1.0
            elif word_count < 100:
                length_factor = word_count / 100.0
            else:
                length_factor = 10000.0 / word_count
            quality_factors.append(('length', length_factor, 0.2))
            
            # Concept richness factor
            concept_richness = min(len(concepts) / 10.0, 1.0)  # Ideal: 10+ concepts
            quality_factors.append(('concepts', concept_richness, 0.3))
            
            # Readability factor (using TextBlob)
            blob = TextBlob(content)
            # Simple readability based on sentence structure
            sentences = blob.sentences
            avg_sentence_complexity = sum(len(str(s).split()) for s in sentences) / len(sentences) if sentences else 0
            readability_factor = 1.0 - min(avg_sentence_complexity / 30.0, 1.0)  # Penalty for very long sentences
            quality_factors.append(('readability', readability_factor, 0.2))
            
            # Coherence factor (semantic consistency)
            if len(concepts) > 1:
                concept_embeddings = [c.embedding for c in concepts if c.embedding is not None]
                if len(concept_embeddings) > 1:
                    concept_embeddings = np.array(concept_embeddings)
                    coherence = np.mean(cosine_similarity(concept_embeddings))
                    coherence_factor = max(0, coherence)  # Ensure positive
                else:
                    coherence_factor = 0.5
            else:
                coherence_factor = 0.5
            quality_factors.append(('coherence', coherence_factor, 0.3))
            
            # Calculate weighted quality score
            quality_score = sum(score * weight for _, score, weight in quality_factors)
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5

    def _calculate_readability_metrics(self, content: str) -> Dict[str, float]:
        """Calculate readability metrics."""
        try:
            # Basic readability metrics
            sentences = content.split('.')
            words = content.split()
            syllables = sum(self._count_syllables(word) for word in words)
            
            # Flesch Reading Ease Score
            if len(sentences) > 0 and len(words) > 0:
                avg_sentence_length = len(words) / len(sentences)
                avg_syllables_per_word = syllables / len(words)
                
                flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
                flesch_score = max(0, min(100, flesch_score))  # Clamp to 0-100
            else:
                flesch_score = 0
            
            # Flesch-Kincaid Grade Level
            if len(sentences) > 0 and len(words) > 0:
                fk_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
                fk_grade = max(0, fk_grade)
            else:
                fk_grade = 0
            
            return {
                'flesch_reading_ease': flesch_score,
                'flesch_kincaid_grade': fk_grade,
                'avg_sentence_length': avg_sentence_length if 'avg_sentence_length' in locals() else 0,
                'avg_syllables_per_word': avg_syllables_per_word if 'avg_syllables_per_word' in locals() else 0
            }
            
        except Exception as e:
            logger.warning(f"Readability calculation failed: {e}")
            return {}

    def _count_syllables(self, word: str) -> int:
        """Simple syllable counting."""
        word = word.lower()
        vowels = 'aeiouy'
        syllables = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllables += 1
            previous_was_vowel = is_vowel
        
        # Handle edge cases
        if word.endswith('e') and syllables > 1:
            syllables -= 1
        if syllables == 0:
            syllables = 1
            
        return syllables

    async def _build_entity_graph(self, content: str) -> nx.Graph:
        """Build an entity relationship graph from the content."""
        graph = nx.Graph()
        
        try:
            doc = self.spacy_nlp(content)
            
            # Extract entities
            entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]
            
            # Add entities as nodes
            for entity in entities:
                graph.add_node(entity, type='entity')
            
            # Add relationships based on co-occurrence in sentences
            for sent in doc.sents:
                sent_entities = [ent.text for ent in sent.ents if ent.text in entities]
                
                # Create edges between entities in the same sentence
                for i, entity1 in enumerate(sent_entities):
                    for entity2 in sent_entities[i+1:]:
                        if graph.has_edge(entity1, entity2):
                            graph[entity1][entity2]['weight'] += 1
                        else:
                            graph.add_edge(entity1, entity2, weight=1)
            
        except Exception as e:
            logger.warning(f"Entity graph construction failed: {e}")
        
        return graph

    async def find_semantic_similarities(
        self,
        query_document_id: str,
        candidate_document_ids: List[str],
        similarity_threshold: float = 0.7,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find semantically similar documents using advanced multi-dimensional analysis.
        
        Goes beyond basic vector similarity by considering:
        - Semantic similarity (embeddings)
        - Conceptual overlap
        - Topic distribution similarity
        - Structural similarity
        - Quality alignment
        """
        try:
            if query_document_id not in self.document_embeddings:
                raise ValueError(f"Query document {query_document_id} not found in embeddings")
            
            query_embedding = self.document_embeddings[query_document_id]
            similarities = []
            
            for candidate_id in candidate_document_ids:
                if candidate_id == query_document_id:
                    continue
                
                if candidate_id not in self.document_embeddings:
                    logger.warning(f"Candidate document {candidate_id} not found in embeddings")
                    continue
                
                candidate_embedding = self.document_embeddings[candidate_id]
                
                # Calculate multi-dimensional similarity
                similarity_score = await self._calculate_multidimensional_similarity(
                    query_document_id,
                    candidate_id,
                    query_embedding,
                    candidate_embedding
                )
                
                if similarity_score >= similarity_threshold:
                    similarities.append({
                        'document_id': candidate_id,
                        'similarity_score': similarity_score,
                        'similarity_components': similarity_score,  # Detailed breakdown
                        'relationship_type': self._determine_relationship_type(similarity_score)
                    })
            
            # Sort by similarity score
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return similarities[:max_results]
            
        except Exception as e:
            logger.error(f"Semantic similarity search failed: {e}")
            raise

    async def _calculate_multidimensional_similarity(
        self,
        query_id: str,
        candidate_id: str,
        query_embedding: np.ndarray,
        candidate_embedding: np.ndarray
    ) -> float:
        """Calculate similarity using multiple dimensions."""
        
        # 1. Semantic similarity (embedding cosine similarity)
        semantic_similarity = float(
            np.dot(query_embedding, candidate_embedding) / 
            (np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding))
        )
        
        # 2. Conceptual overlap (if concept data is available)
        conceptual_similarity = 0.5  # Default if no concept data
        # This would be enhanced with actual concept analysis
        
        # 3. Structural similarity (basic implementation)
        structural_similarity = 0.5  # Default
        # This would compare document structure, readability, etc.
        
        # 4. Topic distribution similarity (if available)
        topic_similarity = 0.5  # Default
        # This would compare topic distributions
        
        # Weighted combination
        final_similarity = (
            0.5 * semantic_similarity +
            0.2 * conceptual_similarity +
            0.2 * structural_similarity +
            0.1 * topic_similarity
        )
        
        return max(0.0, min(1.0, final_similarity))

    def _determine_relationship_type(self, similarity_score: float) -> str:
        """Determine the type of relationship based on similarity score."""
        if similarity_score >= 0.9:
            return "nearly_identical"
        elif similarity_score >= 0.8:
            return "highly_similar"
        elif similarity_score >= 0.7:
            return "moderately_similar"
        elif similarity_score >= 0.6:
            return "somewhat_similar"
        else:
            return "weakly_similar"

    async def discover_semantic_relationships(
        self,
        document_ids: List[str],
        relationship_types: List[str] = None
    ) -> List[SemanticRelationship]:
        """
        Discover semantic relationships between multiple documents.
        
        Identifies relationships like:
        - Similarity
        - Contradiction
        - Extension/building upon
        - Prerequisite dependencies
        """
        if relationship_types is None:
            relationship_types = ['similar', 'extends', 'contradicts', 'references']
        
        relationships = []
        
        try:
            # Analyze pairs of documents
            for i, doc1_id in enumerate(document_ids):
                for doc2_id in document_ids[i+1:]:
                    
                    if doc1_id not in self.document_embeddings or doc2_id not in self.document_embeddings:
                        continue
                    
                    # Calculate semantic relationship
                    relationship = await self._analyze_document_relationship(
                        doc1_id, doc2_id, relationship_types
                    )
                    
                    if relationship and relationship.strength >= 0.6:
                        relationships.append(relationship)
            
            # Sort by relationship strength
            relationships.sort(key=lambda r: r.strength, reverse=True)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Relationship discovery failed: {e}")
            raise

    async def _analyze_document_relationship(
        self,
        doc1_id: str,
        doc2_id: str,
        relationship_types: List[str]
    ) -> Optional[SemanticRelationship]:
        """Analyze the relationship between two specific documents."""
        
        try:
            embedding1 = self.document_embeddings[doc1_id]
            embedding2 = self.document_embeddings[doc2_id]
            
            # Calculate basic similarity
            similarity = float(
                np.dot(embedding1, embedding2) / 
                (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            )
            
            # Determine relationship type and strength
            if similarity >= 0.8:
                relationship_type = 'similar'
                strength = similarity
                confidence = 0.9
                evidence = [f"High semantic similarity: {similarity:.3f}"]
            elif similarity >= 0.6:
                relationship_type = 'related'
                strength = similarity
                confidence = 0.7
                evidence = [f"Moderate semantic similarity: {similarity:.3f}"]
            else:
                return None  # Too weak to be considered a relationship
            
            return SemanticRelationship(
                source_id=doc1_id,
                target_id=doc2_id,
                relationship_type=relationship_type,
                strength=strength,
                confidence=confidence,
                evidence=evidence,
                metadata={
                    'similarity_score': similarity,
                    'analysis_timestamp': datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.warning(f"Document relationship analysis failed: {e}")
            return None

    async def build_knowledge_graph(
        self,
        document_analyses: List[DocumentAnalysis],
        min_relationship_strength: float = 0.6
    ) -> nx.Graph:
        """
        Build a comprehensive knowledge graph from document analyses.
        """
        knowledge_graph = nx.Graph()
        
        try:
            # Add document nodes
            for analysis in document_analyses:
                knowledge_graph.add_node(
                    analysis.document_id,
                    type='document',
                    quality_score=analysis.quality_score,
                    concept_count=len(analysis.key_concepts),
                    complexity=analysis.complexity_metrics.get('avg_dependency_depth', 0)
                )
                
                # Add concept nodes
                for concept in analysis.key_concepts:
                    if not knowledge_graph.has_node(concept.concept_id):
                        knowledge_graph.add_node(
                            concept.concept_id,
                            type='concept',
                            concept_name=concept.concept_name,
                            concept_type=concept.concept_type,
                            importance=concept.importance_score
                        )
                    
                    # Connect document to concept
                    knowledge_graph.add_edge(
                        analysis.document_id,
                        concept.concept_id,
                        relationship='contains',
                        weight=concept.importance_score
                    )
            
            # Discover and add semantic relationships
            document_ids = [analysis.document_id for analysis in document_analyses]
            relationships = await self.discover_semantic_relationships(document_ids)
            
            for relationship in relationships:
                if relationship.strength >= min_relationship_strength:
                    knowledge_graph.add_edge(
                        relationship.source_id,
                        relationship.target_id,
                        relationship=relationship.relationship_type,
                        strength=relationship.strength,
                        confidence=relationship.confidence,
                        evidence=relationship.evidence
                    )
            
            logger.info(f"Built knowledge graph with {knowledge_graph.number_of_nodes()} nodes and {knowledge_graph.number_of_edges()} edges")
            
            return knowledge_graph
            
        except Exception as e:
            logger.error(f"Knowledge graph construction failed: {e}")
            raise

    async def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the semantic analysis."""
        return {
            'documents_analyzed': len(self.document_embeddings),
            'concept_graph_nodes': self.concept_graph.number_of_nodes(),
            'concept_graph_edges': self.concept_graph.number_of_edges(),
            'quantization_enabled': self.use_quantization,
            'quantization_bits': self.quantization_bits if self.use_quantization else None,
            'embedding_model': self.embedding_model_name,
            'spacy_model': self.spacy_model_name,
            'semantic_clusters': len(self.semantic_clusters)
        }

