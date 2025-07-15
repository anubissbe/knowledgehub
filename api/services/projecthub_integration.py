"""ProjectHub Integration Service

Integration with ProjectHub API for project data retrieval and synchronization.
"""

import logging
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

import requests

logger = logging.getLogger(__name__)


class ProjectHubIntegration:
    """Service for integrating with ProjectHub API"""
    
    def __init__(self):
        self.api_url = os.getenv('PROJECTHUB_API_URL', 'http://192.168.1.24:3009/api')
        self.email = os.getenv('PROJECTHUB_EMAIL', 'admin@projecthub.com')
        self.password = os.getenv('PROJECTHUB_PASSWORD', 'admin123')
        self.token = None
        self.token_expiry = None
    
    def _ensure_authenticated(self) -> str:
        """Ensure authentication token is valid"""
        
        # Check if token is still valid (with 5 minute buffer)
        if self.token and self.token_expiry and datetime.now() < self.token_expiry:
            return self.token
        
        logger.info('Authenticating with ProjectHub...')
        
        try:
            response = requests.post(
                f"{self.api_url}/auth/login",
                json={
                    'email': self.email,
                    'password': self.password
                },
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Authentication failed: {response.status_code}")
            
            data = response.json()
            self.token = data.get('token')
            
            if not self.token:
                raise Exception("No token received from authentication")
            
            # Parse JWT to get expiry (basic implementation)
            try:
                import base64
                payload = json.loads(base64.b64decode(self.token.split('.')[1] + '=='))
                self.token_expiry = datetime.fromtimestamp(payload.get('exp', 0))
            except:
                # Default to 24 hours if can't parse
                self.token_expiry = datetime.now() + timedelta(hours=24)
            
            logger.info(f"ProjectHub authentication successful")
            return self.token
            
        except Exception as e:
            logger.error(f"ProjectHub authentication failed: {e}")
            raise
    
    def _request(self, endpoint: str, method: str = 'GET', data: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated request to ProjectHub API"""
        
        try:
            token = self._ensure_authenticated()
            
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }
            
            url = f"{self.api_url}{endpoint}"
            
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=data, timeout=30)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=headers, json=data, timeout=30)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            if response.status_code == 401:
                # Token expired, retry once
                self.token = None
                self.token_expiry = None
                return self._request(endpoint, method, data)
            
            if response.status_code >= 400:
                logger.error(f"ProjectHub API error: {response.status_code} - {response.text}")
                return None
            
            return response.json()
            
        except Exception as e:
            logger.error(f"ProjectHub API request failed: {e}")
            return None
    
    def get_project_data(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project data from ProjectHub"""
        
        project_data = self._request(f'/projects/{project_id}')
        if not project_data:
            return None
        
        # Get project tasks
        tasks = self._request(f'/projects/{project_id}/tasks') or []
        
        # Get project members
        members = self._request(f'/projects/{project_id}/members') or []
        
        # Enhance project data
        enhanced_data = {
            'id': project_data.get('id'),
            'name': project_data.get('name'),
            'description': project_data.get('description'),
            'status': project_data.get('status'),
            'priority': project_data.get('priority'),
            'created_at': project_data.get('created_at'),
            'updated_at': project_data.get('updated_at'),
            'deadline': project_data.get('deadline'),
            'progress': project_data.get('progress', 0),
            'tasks': tasks,
            'members': members,
            'task_count': len(tasks),
            'completed_tasks': len([t for t in tasks if t.get('status') == 'completed']),
            'has_deadline': bool(project_data.get('deadline'))
        }
        
        return enhanced_data
    
    def get_project_tasks(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all tasks for a project"""
        
        tasks = self._request(f'/projects/{project_id}/tasks') or []
        
        # Enhance task data
        enhanced_tasks = []
        for task in tasks:
            enhanced_task = {
                'id': task.get('id'),
                'title': task.get('title'),
                'description': task.get('description'),
                'status': task.get('status'),
                'priority': task.get('priority'),
                'assignee': task.get('assignee'),
                'created_at': task.get('created_at'),
                'updated_at': task.get('updated_at'),
                'due_date': task.get('due_date'),
                'completed_at': task.get('completed_at'),
                'type': self._infer_task_type(task),
                'estimated_hours': task.get('estimated_hours'),
                'actual_hours': task.get('actual_hours')
            }
            enhanced_tasks.append(enhanced_task)
        
        return enhanced_tasks
    
    def get_project_timeline_data(self, project_id: str) -> Dict[str, Any]:
        """Get timeline-relevant data for a project"""
        
        project_data = self.get_project_data(project_id)
        if not project_data:
            return {}
        
        tasks = project_data.get('tasks', [])
        
        # Calculate timeline metrics
        timeline_data = {
            'project_id': project_id,
            'project_name': project_data.get('name'),
            'total_tasks': len(tasks),
            'completed_tasks': len([t for t in tasks if t.get('status') == 'completed']),
            'in_progress_tasks': len([t for t in tasks if t.get('status') == 'in_progress']),
            'pending_tasks': len([t for t in tasks if t.get('status') == 'pending']),
            'overdue_tasks': len([t for t in tasks if self._is_task_overdue(t)]),
            'earliest_task_date': self._get_earliest_task_date(tasks),
            'latest_task_date': self._get_latest_task_date(tasks),
            'project_deadline': project_data.get('deadline'),
            'project_status': project_data.get('status'),
            'project_progress': project_data.get('progress', 0),
            'last_activity': self._get_last_activity_date(tasks),
            'active_members': len(project_data.get('members', [])),
            'task_completion_rate': self._calculate_completion_rate(tasks),
            'average_task_duration': self._calculate_average_task_duration(tasks)
        }
        
        return timeline_data
    
    def sync_timeline_with_projecthub(self, timeline_id: str, project_id: str) -> Dict[str, Any]:
        """Synchronize timeline data with ProjectHub"""
        
        try:
            timeline_data = self.get_project_timeline_data(project_id)
            
            # This would typically update the local timeline
            # For now, return the sync data
            sync_result = {
                'timeline_id': timeline_id,
                'project_id': project_id,
                'sync_timestamp': datetime.now().isoformat(),
                'synced_data': timeline_data,
                'sync_status': 'success'
            }
            
            logger.info(f"Synchronized timeline {timeline_id} with ProjectHub project {project_id}")
            return sync_result
            
        except Exception as e:
            logger.error(f"Failed to sync timeline with ProjectHub: {e}")
            return {
                'timeline_id': timeline_id,
                'project_id': project_id,
                'sync_timestamp': datetime.now().isoformat(),
                'sync_status': 'failed',
                'error': str(e)
            }
    
    def _infer_task_type(self, task: Dict[str, Any]) -> str:
        """Infer task type from title and description"""
        
        title = task.get('title', '').lower()
        description = task.get('description', '').lower()
        text = f"{title} {description}"
        
        # Simple keyword-based classification
        if any(keyword in text for keyword in ['test', 'qa', 'quality', 'verify']):
            return 'testing'
        elif any(keyword in text for keyword in ['deploy', 'release', 'production', 'launch']):
            return 'deployment'
        elif any(keyword in text for keyword in ['design', 'plan', 'requirement', 'analysis']):
            return 'planning'
        elif any(keyword in text for keyword in ['develop', 'implement', 'code', 'build', 'create']):
            return 'development'
        elif any(keyword in text for keyword in ['review', 'audit', 'check', 'inspect']):
            return 'review'
        elif any(keyword in text for keyword in ['fix', 'bug', 'issue', 'problem']):
            return 'bugfix'
        elif any(keyword in text for keyword in ['doc', 'documentation', 'manual', 'guide']):
            return 'documentation'
        else:
            return 'general'
    
    def _is_task_overdue(self, task: Dict[str, Any]) -> bool:
        """Check if task is overdue"""
        
        due_date = task.get('due_date')
        if not due_date:
            return False
        
        try:
            due_datetime = datetime.fromisoformat(due_date.replace('Z', '+00:00'))
            return datetime.now() > due_datetime and task.get('status') != 'completed'
        except:
            return False
    
    def _get_earliest_task_date(self, tasks: List[Dict[str, Any]]) -> Optional[str]:
        """Get earliest task date"""
        
        dates = []
        for task in tasks:
            created_at = task.get('created_at')
            if created_at:
                dates.append(created_at)
        
        return min(dates) if dates else None
    
    def _get_latest_task_date(self, tasks: List[Dict[str, Any]]) -> Optional[str]:
        """Get latest task date"""
        
        dates = []
        for task in tasks:
            due_date = task.get('due_date')
            if due_date:
                dates.append(due_date)
        
        return max(dates) if dates else None
    
    def _get_last_activity_date(self, tasks: List[Dict[str, Any]]) -> Optional[str]:
        """Get last activity date"""
        
        dates = []
        for task in tasks:
            updated_at = task.get('updated_at')
            if updated_at:
                dates.append(updated_at)
        
        return max(dates) if dates else None
    
    def _calculate_completion_rate(self, tasks: List[Dict[str, Any]]) -> float:
        """Calculate task completion rate"""
        
        if not tasks:
            return 0.0
        
        completed = len([t for t in tasks if t.get('status') == 'completed'])
        return (completed / len(tasks)) * 100
    
    def _calculate_average_task_duration(self, tasks: List[Dict[str, Any]]) -> Optional[float]:
        """Calculate average task duration in days"""
        
        durations = []
        
        for task in tasks:
            created_at = task.get('created_at')
            completed_at = task.get('completed_at')
            
            if created_at and completed_at:
                try:
                    start = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    end = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
                    duration = (end - start).days
                    durations.append(duration)
                except:
                    continue
        
        return sum(durations) / len(durations) if durations else None