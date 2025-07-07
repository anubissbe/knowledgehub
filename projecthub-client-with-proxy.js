#!/usr/bin/env node

/**
 * ProjectHub Client with Proxy Support
 * 
 * This version uses the proxy server to work around the broken PUT endpoint
 */

class ProjectHubClient {
  constructor(useProxy = true) {
    // Use proxy by default to fix PUT endpoint issues
    this.apiUrl = useProxy 
      ? 'http://localhost:3109/api'  // Proxy endpoint
      : 'http://192.168.1.24:3009/api';  // Direct endpoint
    
    this.email = 'admin@projecthub.com';
    this.password = 'admin123';
    this.token = null;
    this.tokenExpiry = null;
    this.useProxy = useProxy;
    
    if (useProxy) {
      console.log('Using ProjectHub Proxy to fix PUT endpoint issues');
    }
  }

  async ensureAuthenticated() {
    // Check if token is still valid (with 5 minute buffer)
    if (this.token && this.tokenExpiry && new Date() < new Date(this.tokenExpiry - 5 * 60 * 1000)) {
      return this.token;
    }

    console.log('Authenticating with ProjectHub...');
    const response = await fetch(`${this.apiUrl}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: this.email,
        password: this.password
      })
    });

    if (!response.ok) {
      // Authentication might not be required for proxy
      if (this.useProxy) {
        console.log('Proxy mode: Authentication not required');
        return null;
      }
      throw new Error(`Authentication failed: ${response.statusText}`);
    }

    const data = await response.json();
    this.token = data.token;

    // Parse JWT to get expiry
    try {
      const payload = JSON.parse(atob(this.token.split('.')[1]));
      this.tokenExpiry = new Date(payload.exp * 1000);
      console.log(`Token valid until: ${this.tokenExpiry.toISOString()}`);
    } catch (e) {
      // Default to 24 hours if can't parse
      this.tokenExpiry = new Date(Date.now() + 24 * 60 * 60 * 1000);
    }

    return this.token;
  }

  async request(endpoint, options = {}) {
    // For proxy mode, skip authentication
    const token = this.useProxy ? null : await this.ensureAuthenticated();

    const headers = {
      'Content-Type': 'application/json',
      ...options.headers
    };
    
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(`${this.apiUrl}${endpoint}`, {
      ...options,
      headers
    });

    // Handle token expiry (only for direct mode)
    if (!this.useProxy && response.status === 401) {
      console.log('Token expired, re-authenticating...');
      this.token = null;
      const newToken = await this.ensureAuthenticated();

      // Retry request with new token
      const retryResponse = await fetch(`${this.apiUrl}${endpoint}`, {
        ...options,
        headers: {
          'Authorization': `Bearer ${newToken}`,
          'Content-Type': 'application/json',
          ...options.headers
        }
      });

      if (!retryResponse.ok) {
        throw new Error(`API request failed: ${retryResponse.statusText}`);
      }

      return retryResponse.json();
    }

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API request failed: ${response.statusText} - ${errorText}`);
    }

    return response.json();
  }

  // Task Management Methods
  async listTasks(projectId = null, status = null) {
    let query = '';
    const params = new URLSearchParams();
    if (projectId) params.append('projectId', projectId);
    if (status) params.append('status', status);
    if (params.toString()) query = `?${params.toString()}`;

    return this.request(`/tasks${query}`);
  }

  async getMyTasks() {
    // Get all in-progress tasks
    const inProgress = await this.listTasks(null, 'in_progress');
    const pending = await this.listTasks(null, 'pending');

    return {
      inProgress,
      pending,
      total: inProgress.length + pending.length
    };
  }

  async createTask(data) {
    return this.request('/tasks', {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  async updateTask(taskId, data) {
    const response = await this.request(`/tasks/${taskId}`, {
      method: 'PUT',
      body: JSON.stringify(data)
    });
    
    // If using proxy, check for workaround metadata
    if (response.update_method === 'create_new') {
      console.log(`Note: Task updated via proxy workaround (new task created)`);
    }
    
    return response;
  }

  async startTask(taskId) {
    return this.updateTask(taskId, {
      status: 'in_progress',
      started_at: new Date().toISOString(),
      notes: 'Started working on this task'
    });
  }

  async updateTaskProgress(taskId, progress, notes) {
    return this.updateTask(taskId, {
      progress,
      notes,
      updated_at: new Date().toISOString()
    });
  }

  async completeTask(taskId, notes = 'Task completed') {
    return this.updateTask(taskId, {
      status: 'completed',
      progress: 100,
      completed_at: new Date().toISOString(),
      notes
    });
  }

  async listProjects() {
    return this.request('/projects');
  }

  async createProject(data) {
    return this.request('/projects', {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  // Helper method to display task status
  formatTaskStatus(task) {
    const status = {
      pending: '⏳',
      in_progress: '🔄',
      completed: '✅',
      cancelled: '❌'
    };

    return `${status[task.status] || '❓'} [${task.id}] ${task.title} (${task.progress || 0}%)`;
  }
}

// Initialize client with proxy by default
const projectHub = new ProjectHubClient(true);

// Helper functions for common operations
async function checkMyTasks() {
  try {
    const tasks = await projectHub.getMyTasks();

    console.log('\n📋 Your ProjectHub Tasks:');

    if (tasks.inProgress.length > 0) {
      console.log('\n🔄 In Progress:');
      tasks.inProgress.forEach(task => {
        console.log(`  ${projectHub.formatTaskStatus(task)}`);
        if (task.notes) console.log(`     Last update: ${task.notes}`);
      });
    }

    if (tasks.pending.length > 0) {
      console.log('\n⏳ Pending:');
      tasks.pending.slice(0, 5).forEach(task => {
        console.log(`  ${projectHub.formatTaskStatus(task)}`);
      });
      if (tasks.pending.length > 5) {
        console.log(`  ... and ${tasks.pending.length - 5} more pending tasks`);
      }
    }

    if (tasks.total === 0) {
      console.log('  No active tasks found');
    }

    return tasks;
  } catch (error) {
    console.error('Failed to fetch tasks:', error.message);
    return { inProgress: [], pending: [], total: 0 };
  }
}

async function createTaskForWork(title, description, estimateHours = 2) {
  try {
    // Find or create a default project
    const projects = await projectHub.listProjects();
    let project = projects.find(p => p.name === 'Development Tasks');

    if (!project) {
      project = await projectHub.createProject({
        name: 'Development Tasks',
        description: 'General development and maintenance tasks',
        status: 'active'
      });
    }

    // Create the task
    const task = await projectHub.createTask({
      project_id: project.id,
      title,
      description,
      status: 'pending',
      priority: 'medium',
      estimate_hours: estimateHours
    });

    console.log(`\n✅ Created task: ${task.title} (ID: ${task.id})`);
    return task;
  } catch (error) {
    console.error('Failed to create task:', error.message);
    return null;
  }
}

// Test the proxy functionality
async function testProxy() {
  console.log('\n🧪 Testing ProjectHub Proxy Functionality\n');
  
  try {
    // 1. Get tasks
    console.log('1. Fetching tasks...');
    const tasks = await projectHub.listTasks();
    console.log(`   ✅ Found ${tasks.length} tasks`);
    
    if (tasks.length === 0) {
      console.log('   No tasks available for testing');
      return;
    }
    
    // 2. Find a pending task to test with
    const pendingTask = tasks.find(t => t.status === 'pending');
    if (!pendingTask) {
      console.log('   No pending tasks available for testing');
      return;
    }
    
    console.log(`\n2. Testing PUT endpoint with task: ${pendingTask.title}`);
    console.log(`   Current status: ${pendingTask.status}`);
    
    // 3. Update task status
    console.log('\n3. Updating task status to "completed"...');
    const updatedTask = await projectHub.updateTask(pendingTask.id, {
      status: 'completed',
      notes: 'Completed via proxy test'
    });
    
    console.log(`   ✅ Update successful!`);
    console.log(`   New task ID: ${updatedTask.id}`);
    console.log(`   Status: ${updatedTask.status}`);
    if (updatedTask.update_method) {
      console.log(`   Update method: ${updatedTask.update_method}`);
    }
    
    console.log('\n✅ Proxy is working correctly!');
    
  } catch (error) {
    console.error('\n❌ Proxy test failed:', error.message);
  }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { 
    projectHub, 
    checkMyTasks, 
    createTaskForWork,
    ProjectHubClient,
    testProxy
  };
}

// If run directly, perform a test
if (require.main === module) {
  testProxy().catch(console.error);
}