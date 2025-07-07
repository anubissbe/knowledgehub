"""HTTP server for scheduler management"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from aiohttp import web, web_request
from aiohttp.web_response import Response

logger = logging.getLogger(__name__)


class SchedulerHTTPServer:
    def __init__(self, scheduler_instance, port: int = 8080):
        self.scheduler = scheduler_instance
        self.port = port
        self.app = web.Application()
        self.setup_routes()
        
    def setup_routes(self):
        """Setup HTTP routes for scheduler management"""
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/jobs', self.get_jobs)
        self.app.router.add_post('/manual-refresh', self.manual_refresh)
        self.app.router.add_post('/jobs/{job_id}/pause', self.pause_job)
        self.app.router.add_post('/jobs/{job_id}/resume', self.resume_job)
        self.app.router.add_put('/config', self.update_config)
        self.app.router.add_get('/status', self.get_status)
        
    async def health_check(self, request: web_request.Request) -> Response:
        """Health check endpoint"""
        try:
            status = {
                "status": "healthy",
                "scheduler_running": self.scheduler.scheduler.running if self.scheduler.scheduler else False,
                "jobs_count": len(self.scheduler.scheduler.get_jobs()) if self.scheduler.scheduler else 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            return web.json_response(status)
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return web.json_response(
                {"status": "error", "error": str(e)}, 
                status=500
            )
    
    async def get_jobs(self, request: web_request.Request) -> Response:
        """Get all scheduled jobs"""
        try:
            if not self.scheduler.scheduler or not self.scheduler.scheduler.running:
                return web.json_response({"jobs": [], "message": "Scheduler not running"})
            
            jobs = []
            for job in self.scheduler.scheduler.get_jobs():
                job_info = {
                    "id": job.id,
                    "name": job.name,
                    "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                    "trigger": str(job.trigger),
                    "function": f"{job.func.__module__}.{job.func.__name__}",
                    "args": list(job.args) if job.args else [],
                    "kwargs": dict(job.kwargs) if job.kwargs else {}
                }
                jobs.append(job_info)
            
            return web.json_response({
                "jobs": jobs,
                "count": len(jobs),
                "scheduler_running": True
            })
            
        except Exception as e:
            logger.error(f"Error getting jobs: {e}")
            return web.json_response(
                {"error": str(e), "jobs": []}, 
                status=500
            )
    
    async def manual_refresh(self, request: web_request.Request) -> Response:
        """Trigger manual refresh"""
        try:
            data = await request.json()
            source_id = data.get("source_id")
            
            # Call the scheduler's manual refresh method
            result = await self.scheduler.manual_refresh(source_id)
            
            return web.json_response({
                "success": True,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in manual refresh: {e}")
            return web.json_response(
                {"success": False, "error": str(e)}, 
                status=500
            )
    
    async def pause_job(self, request: web_request.Request) -> Response:
        """Pause a scheduled job"""
        try:
            job_id = request.match_info['job_id']
            
            if not self.scheduler.scheduler or not self.scheduler.scheduler.running:
                return web.json_response(
                    {"success": False, "error": "Scheduler not running"}, 
                    status=503
                )
            
            job = self.scheduler.scheduler.get_job(job_id)
            if not job:
                return web.json_response(
                    {"success": False, "error": f"Job {job_id} not found"}, 
                    status=404
                )
            
            self.scheduler.scheduler.pause_job(job_id)
            
            return web.json_response({
                "success": True,
                "message": f"Job {job_id} paused",
                "job_id": job_id
            })
            
        except Exception as e:
            logger.error(f"Error pausing job {job_id}: {e}")
            return web.json_response(
                {"success": False, "error": str(e)}, 
                status=500
            )
    
    async def resume_job(self, request: web_request.Request) -> Response:
        """Resume a paused job"""
        try:
            job_id = request.match_info['job_id']
            
            if not self.scheduler.scheduler or not self.scheduler.scheduler.running:
                return web.json_response(
                    {"success": False, "error": "Scheduler not running"}, 
                    status=503
                )
            
            job = self.scheduler.scheduler.get_job(job_id)
            if not job:
                return web.json_response(
                    {"success": False, "error": f"Job {job_id} not found"}, 
                    status=404
                )
            
            self.scheduler.scheduler.resume_job(job_id)
            
            return web.json_response({
                "success": True,
                "message": f"Job {job_id} resumed",
                "job_id": job_id
            })
            
        except Exception as e:
            logger.error(f"Error resuming job {job_id}: {e}")
            return web.json_response(
                {"success": False, "error": str(e)}, 
                status=500
            )
    
    async def update_config(self, request: web_request.Request) -> Response:
        """Update scheduler configuration"""
        try:
            config = await request.json()
            
            # For now, return the current config
            # In a full implementation, this would update environment variables
            # or restart the scheduler with new settings
            
            current_config = {
                "scheduler_enabled": getattr(self.scheduler, 'enabled', True),
                "refresh_schedule": getattr(self.scheduler, 'refresh_schedule', "0 2 * * 0"),
                "refresh_batch_size": getattr(self.scheduler, 'batch_size', 2),
                "refresh_delay_seconds": getattr(self.scheduler, 'delay_seconds', 300)
            }
            
            return web.json_response({
                "success": True,
                "message": "Configuration noted (restart required for changes)",
                "current_config": current_config,
                "requested_config": config
            })
            
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return web.json_response(
                {"success": False, "error": str(e)}, 
                status=500
            )
    
    async def get_status(self, request: web_request.Request) -> Response:
        """Get detailed scheduler status"""
        try:
            if not self.scheduler.scheduler:
                return web.json_response({
                    "status": "not_initialized",
                    "running": False,
                    "jobs": []
                })
            
            jobs = []
            if self.scheduler.scheduler.running:
                for job in self.scheduler.scheduler.get_jobs():
                    jobs.append({
                        "id": job.id,
                        "name": job.name,
                        "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                        "paused": job.next_run_time is None and job.trigger is not None
                    })
            
            status = {
                "status": "running" if self.scheduler.scheduler.running else "stopped",
                "running": self.scheduler.scheduler.running,
                "jobs": jobs,
                "job_count": len(jobs),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return web.json_response(status)
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return web.json_response(
                {"status": "error", "error": str(e)}, 
                status=500
            )
    
    async def start_server(self):
        """Start the HTTP server"""
        try:
            runner = web.AppRunner(self.app)
            await runner.setup()
            
            site = web.TCPSite(runner, '0.0.0.0', self.port)
            await site.start()
            
            logger.info(f"Scheduler HTTP server started on port {self.port}")
            
            return runner
            
        except Exception as e:
            logger.error(f"Failed to start HTTP server: {e}")
            raise
    
    async def stop_server(self, runner):
        """Stop the HTTP server"""
        try:
            await runner.cleanup()
            logger.info("Scheduler HTTP server stopped")
        except Exception as e:
            logger.error(f"Error stopping HTTP server: {e}")