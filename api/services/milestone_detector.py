"""Milestone Detection Service

Automatically identifies and creates milestones based on project patterns,
task activities, and integration with external systems like ProjectHub.
"""

import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID

from sqlalchemy.orm import Session

from .project_timeline_service import ProjectTimelineService
from ..models.project_timeline import ProjectTimeline, MilestoneType, MilestoneStatus
from .projecthub_integration import ProjectHubIntegration

logger = logging.getLogger(__name__)


class MilestonePattern:
    """Represents a milestone detection pattern"""
    
    def __init__(
        self,
        name: str,
        pattern: str,
        milestone_type: str,
        confidence_weight: float,
        description_template: str,
        keywords: List[str]
    ):
        self.name = name
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.milestone_type = milestone_type
        self.confidence_weight = confidence_weight
        self.description_template = description_template
        self.keywords = [k.lower() for k in keywords]
    
    def matches(self, text: str) -> bool:
        """Check if pattern matches text"""
        return bool(self.pattern.search(text))
    
    def get_confidence(self, text: str, context: Dict[str, Any]) -> float:
        """Calculate confidence score for this pattern"""
        base_confidence = self.confidence_weight
        
        # Boost confidence if keywords are present
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in self.keywords if keyword in text_lower)
        keyword_boost = min(0.3, keyword_matches * 0.1)
        
        # Context-based adjustments
        context_boost = 0.0
        if context.get('task_count', 0) > 5:
            context_boost += 0.1
        if context.get('has_deadline', False):
            context_boost += 0.1
        
        return min(1.0, base_confidence + keyword_boost + context_boost)


class MilestoneDetector:
    """Service for automatic milestone detection"""
    
    def __init__(self, db: Session):
        self.db = db
        self.timeline_service = ProjectTimelineService(db)
        self.projecthub_integration = ProjectHubIntegration()
        
        # Initialize detection patterns
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> List[MilestonePattern]:
        """Initialize milestone detection patterns"""
        
        return [
            MilestonePattern(
                name="Project Kickoff",
                pattern=r"(kickoff|start|begin|launch|initiate)",
                milestone_type=MilestoneType.PLANNING.value,
                confidence_weight=0.8,
                description_template="Project kickoff and initialization",
                keywords=["kickoff", "start", "begin", "launch", "initiate", "setup"]
            ),
            MilestonePattern(
                name="Requirements Complete",
                pattern=r"(requirements?|specs?|specification).*?(complete|done|finish)",
                milestone_type=MilestoneType.PLANNING.value,
                confidence_weight=0.7,
                description_template="Requirements and specifications completed",
                keywords=["requirements", "specs", "specification", "analysis", "design"]
            ),
            MilestonePattern(
                name="Development Start",
                pattern=r"(development|coding|implementation).*?(start|begin)",
                milestone_type=MilestoneType.DEVELOPMENT.value,
                confidence_weight=0.8,
                description_template="Development phase started",
                keywords=["development", "coding", "implementation", "build", "create"]
            ),
            MilestonePattern(
                name="Alpha Release",
                pattern=r"alpha.*?(release|version|build)",
                milestone_type=MilestoneType.DEVELOPMENT.value,
                confidence_weight=0.9,
                description_template="Alpha release completed",
                keywords=["alpha", "release", "version", "prototype", "mvp"]
            ),
            MilestonePattern(
                name="Beta Release",
                pattern=r"beta.*?(release|version|build)",
                milestone_type=MilestoneType.TESTING.value,
                confidence_weight=0.9,
                description_template="Beta release completed",
                keywords=["beta", "release", "testing", "preview", "candidate"]
            ),
            MilestonePattern(
                name="Testing Complete",
                pattern=r"(testing|qa|quality).*?(complete|done|finish)",
                milestone_type=MilestoneType.TESTING.value,
                confidence_weight=0.7,
                description_template="Testing phase completed",
                keywords=["testing", "qa", "quality", "validation", "verification"]
            ),
            MilestonePattern(
                name="Code Review",
                pattern=r"(code.*?review|review.*?code|peer.*?review)",
                milestone_type=MilestoneType.REVIEW.value,
                confidence_weight=0.6,
                description_template="Code review completed",
                keywords=["review", "code", "peer", "audit", "inspection"]
            ),
            MilestonePattern(
                name="Documentation Complete",
                pattern=r"(documentation|docs|manual).*?(complete|done|finish)",
                milestone_type=MilestoneType.DEVELOPMENT.value,
                confidence_weight=0.5,
                description_template="Documentation completed",
                keywords=["documentation", "docs", "manual", "guide", "readme"]
            ),
            MilestonePattern(
                name="Deployment Ready",
                pattern=r"(deployment|deploy|release).*?(ready|prepared)",
                milestone_type=MilestoneType.DEPLOYMENT.value,
                confidence_weight=0.8,
                description_template="Ready for deployment",
                keywords=["deployment", "deploy", "release", "production", "live"]
            ),
            MilestonePattern(
                name="Go Live",
                pattern=r"(go.*?live|production|launch|release)",
                milestone_type=MilestoneType.RELEASE.value,
                confidence_weight=0.9,
                description_template="Production release completed",
                keywords=["live", "production", "launch", "release", "public"]
            ),
            MilestonePattern(
                name="Project Complete",
                pattern=r"(project.*?complete|complete.*?project|finish|done|delivered)",
                milestone_type=MilestoneType.RELEASE.value,
                confidence_weight=0.8,
                description_template="Project completed",
                keywords=["complete", "finish", "done", "delivered", "closed"]
            )
        ]
    
    def detect_milestones_from_text(
        self,
        timeline_id: UUID,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Detect milestones from text content"""
        
        context = context or {}
        detected_milestones = []
        
        for pattern in self.patterns:
            if pattern.matches(text):
                confidence = pattern.get_confidence(text, context)
                
                if confidence > 0.4:  # Minimum confidence threshold
                    milestone_data = {
                        'name': pattern.name,
                        'description': pattern.description_template,
                        'milestone_type': pattern.milestone_type,
                        'confidence': confidence,
                        'pattern_name': pattern.name,
                        'detected_from': 'text_analysis',
                        'source_text': text[:200]  # First 200 characters
                    }
                    
                    detected_milestones.append(milestone_data)
        
        # Sort by confidence (highest first)
        detected_milestones.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detected_milestones
    
    def detect_milestones_from_projecthub(
        self,
        timeline_id: UUID,
        project_id: str
    ) -> List[Dict[str, Any]]:
        """Detect milestones from ProjectHub integration"""
        
        try:
            # Get project data from ProjectHub
            project_data = self.projecthub_integration.get_project_data(project_id)
            if not project_data:
                return []
            
            detected_milestones = []
            
            # Analyze tasks for milestone patterns
            tasks = project_data.get('tasks', [])
            
            # Group tasks by type/category
            task_groups = self._group_tasks_by_type(tasks)
            
            for group_type, group_tasks in task_groups.items():
                if len(group_tasks) >= 3:  # Minimum tasks for a milestone
                    milestone_data = self._create_milestone_from_task_group(
                        group_type, group_tasks, project_data
                    )
                    if milestone_data:
                        detected_milestones.append(milestone_data)
            
            # Detect deadline-based milestones
            deadline_milestones = self._detect_deadline_milestones(project_data)
            detected_milestones.extend(deadline_milestones)
            
            return detected_milestones
            
        except Exception as e:
            logger.error(f"Error detecting milestones from ProjectHub: {e}")
            return []
    
    def detect_milestones_from_activity(
        self,
        timeline_id: UUID,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """Detect milestones from project activity patterns"""
        
        timeline = self.timeline_service.get_project_timeline_by_uuid(timeline_id)
        if not timeline:
            return []
        
        detected_milestones = []
        
        # Analyze progress snapshots for significant events
        snapshots = self.timeline_service.get_progress_history(timeline_id, limit=100)
        
        # Look for significant progress jumps
        for i in range(len(snapshots) - 1):
            current = snapshots[i]
            previous = snapshots[i + 1]
            
            progress_jump = current.progress_percentage - previous.progress_percentage
            
            if progress_jump >= 20:  # Significant progress jump
                milestone_data = {
                    'name': f"Significant Progress - {progress_jump:.1f}%",
                    'description': f"Major progress milestone detected ({progress_jump:.1f}% increase)",
                    'milestone_type': MilestoneType.DEVELOPMENT.value,
                    'confidence': min(0.8, progress_jump / 50),
                    'pattern_name': 'progress_jump',
                    'detected_from': 'activity_analysis',
                    'planned_date': current.snapshot_date,
                    'progress_jump': progress_jump
                }
                
                detected_milestones.append(milestone_data)
        
        # Detect velocity changes
        velocity_milestones = self._detect_velocity_milestones(snapshots)
        detected_milestones.extend(velocity_milestones)
        
        return detected_milestones
    
    def auto_create_milestones(
        self,
        timeline_id: UUID,
        min_confidence: float = 0.6,
        max_milestones: int = 20
    ) -> List[Dict[str, Any]]:
        """Automatically create milestones based on all detection methods"""
        
        timeline = self.timeline_service.get_project_timeline_by_uuid(timeline_id)
        if not timeline:
            return []
        
        all_detected = []
        
        # Detect from project description
        if timeline.project_description:
            text_milestones = self.detect_milestones_from_text(
                timeline_id, timeline.project_description
            )
            all_detected.extend(text_milestones)
        
        # Detect from ProjectHub if integrated
        if timeline.external_references.get('projecthub_id'):
            projecthub_milestones = self.detect_milestones_from_projecthub(
                timeline_id, timeline.external_references['projecthub_id']
            )
            all_detected.extend(projecthub_milestones)
        
        # Detect from activity
        activity_milestones = self.detect_milestones_from_activity(timeline_id)
        all_detected.extend(activity_milestones)
        
        # Remove duplicates and filter by confidence
        unique_milestones = self._deduplicate_milestones(all_detected)
        high_confidence_milestones = [
            m for m in unique_milestones if m['confidence'] >= min_confidence
        ]
        
        # Limit number of milestones
        milestones_to_create = high_confidence_milestones[:max_milestones]
        
        # Create milestones
        created_milestones = []
        for milestone_data in milestones_to_create:
            try:
                milestone = self.timeline_service.create_milestone(
                    timeline_id=timeline_id,
                    name=milestone_data['name'],
                    description=milestone_data['description'],
                    milestone_type=milestone_data['milestone_type'],
                    planned_date=milestone_data.get('planned_date'),
                    auto_detected=True,
                    detection_confidence=milestone_data['confidence'],
                    detection_metadata={
                        'pattern_name': milestone_data.get('pattern_name'),
                        'detected_from': milestone_data.get('detected_from'),
                        'source_data': milestone_data
                    }
                )
                
                if milestone:
                    created_milestones.append({
                        'milestone_id': str(milestone.id),
                        'name': milestone.name,
                        'confidence': milestone.detection_confidence,
                        'type': milestone.milestone_type
                    })
                    
            except Exception as e:
                logger.error(f"Error creating auto-detected milestone: {e}")
        
        logger.info(f"Auto-created {len(created_milestones)} milestones for timeline {timeline_id}")
        return created_milestones
    
    def _group_tasks_by_type(self, tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group tasks by type or category"""
        
        groups = {}
        
        for task in tasks:
            task_type = task.get('type', 'general')
            title = task.get('title', '').lower()
            
            # Categorize based on title keywords
            if any(keyword in title for keyword in ['test', 'qa', 'quality']):
                task_type = 'testing'
            elif any(keyword in title for keyword in ['deploy', 'release', 'production']):
                task_type = 'deployment'
            elif any(keyword in title for keyword in ['design', 'plan', 'requirement']):
                task_type = 'planning'
            elif any(keyword in title for keyword in ['develop', 'implement', 'code', 'build']):
                task_type = 'development'
            elif any(keyword in title for keyword in ['review', 'audit', 'check']):
                task_type = 'review'
            
            if task_type not in groups:
                groups[task_type] = []
            groups[task_type].append(task)
        
        return groups
    
    def _create_milestone_from_task_group(
        self,
        group_type: str,
        tasks: List[Dict[str, Any]],
        project_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create milestone from task group"""
        
        if not tasks:
            return None
        
        # Map group types to milestone types
        type_mapping = {
            'planning': MilestoneType.PLANNING.value,
            'development': MilestoneType.DEVELOPMENT.value,
            'testing': MilestoneType.TESTING.value,
            'deployment': MilestoneType.DEPLOYMENT.value,
            'review': MilestoneType.REVIEW.value
        }
        
        milestone_type = type_mapping.get(group_type, MilestoneType.CUSTOM.value)
        
        # Calculate confidence based on task count and completion
        confidence = min(0.9, 0.3 + (len(tasks) * 0.1))
        
        # Find earliest and latest task dates
        task_dates = []
        for task in tasks:
            if task.get('due_date'):
                task_dates.append(task['due_date'])
        
        planned_date = None
        if task_dates:
            # Use latest date as milestone date
            planned_date = max(task_dates)
        
        return {
            'name': f"{group_type.title()} Phase Complete",
            'description': f"Completion of {group_type} phase ({len(tasks)} tasks)",
            'milestone_type': milestone_type,
            'confidence': confidence,
            'pattern_name': f'task_group_{group_type}',
            'detected_from': 'projecthub_tasks',
            'planned_date': planned_date,
            'task_count': len(tasks)
        }
    
    def _detect_deadline_milestones(self, project_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect milestones based on project deadlines"""
        
        milestones = []
        
        # Check for project deadline
        if project_data.get('deadline'):
            milestones.append({
                'name': "Project Deadline",
                'description': "Project completion deadline",
                'milestone_type': MilestoneType.RELEASE.value,
                'confidence': 0.8,
                'pattern_name': 'project_deadline',
                'detected_from': 'projecthub_deadline',
                'planned_date': project_data['deadline']
            })
        
        # Check for phase deadlines
        phases = project_data.get('phases', [])
        for phase in phases:
            if phase.get('deadline'):
                milestones.append({
                    'name': f"{phase.get('name', 'Phase')} Deadline",
                    'description': f"Deadline for {phase.get('name', 'phase')}",
                    'milestone_type': MilestoneType.CUSTOM.value,
                    'confidence': 0.7,
                    'pattern_name': 'phase_deadline',
                    'detected_from': 'projecthub_phases',
                    'planned_date': phase['deadline']
                })
        
        return milestones
    
    def _detect_velocity_milestones(self, snapshots: List[Any]) -> List[Dict[str, Any]]:
        """Detect milestones based on velocity changes"""
        
        milestones = []
        
        # Look for significant velocity changes
        for i in range(len(snapshots) - 5):  # Need at least 5 snapshots
            current_group = snapshots[i:i+3]
            previous_group = snapshots[i+3:i+6]
            
            current_avg_velocity = sum(s.velocity for s in current_group) / len(current_group)
            previous_avg_velocity = sum(s.velocity for s in previous_group) / len(previous_group)
            
            if previous_avg_velocity > 0:
                velocity_change = (current_avg_velocity - previous_avg_velocity) / previous_avg_velocity
                
                if velocity_change > 0.5:  # 50% velocity increase
                    milestones.append({
                        'name': "Velocity Increase",
                        'description': f"Significant velocity increase detected ({velocity_change:.1%})",
                        'milestone_type': MilestoneType.DEVELOPMENT.value,
                        'confidence': min(0.7, velocity_change),
                        'pattern_name': 'velocity_increase',
                        'detected_from': 'velocity_analysis',
                        'planned_date': current_group[0].snapshot_date,
                        'velocity_change': velocity_change
                    })
        
        return milestones
    
    def _deduplicate_milestones(self, milestones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate milestones based on similarity"""
        
        unique_milestones = []
        seen_names = set()
        
        # Sort by confidence (highest first)
        sorted_milestones = sorted(milestones, key=lambda x: x['confidence'], reverse=True)
        
        for milestone in sorted_milestones:
            name_key = milestone['name'].lower().replace(' ', '_')
            
            # Simple deduplication based on name similarity
            if name_key not in seen_names:
                unique_milestones.append(milestone)
                seen_names.add(name_key)
        
        return unique_milestones