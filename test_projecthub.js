const fetch = require('node-fetch');

async function testProjectHub() {
    const apiUrl = 'http://192.168.1.24:3009/api';
    
    console.log('Testing ProjectHub API...\n');
    
    // Test 1: Check if API is responding
    try {
        const healthResponse = await fetch(`${apiUrl}/health`);
        console.log(`Health endpoint status: ${healthResponse.status}`);
        if (healthResponse.status !== 404) {
            const healthData = await healthResponse.text();
            console.log('Health response:', healthData);
        }
    } catch (error) {
        console.log('Health check failed:', error.message);
    }
    
    // Test 2: Try to login with different credentials
    console.log('\nTesting authentication...');
    const credentials = [
        { email: 'admin@projecthub.com', password: 'admin123' },
        { email: 'admin@example.com', password: 'admin' },
        { email: 'admin', password: 'admin' }
    ];
    
    for (const cred of credentials) {
        try {
            const response = await fetch(`${apiUrl}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(cred)
            });
            const data = await response.json();
            console.log(`Login with ${cred.email}: ${response.status}`, data);
            
            if (data.token) {
                console.log('\nAuthentication successful! Testing task update...');
                
                // First, list tasks to get a valid ID
                const listResponse = await fetch(`${apiUrl}/tasks`, {
                    headers: {
                        'Authorization': `Bearer ${data.token}`,
                        'Content-Type': 'application/json'
                    }
                });
                const tasks = await listResponse.json();
                console.log('Tasks response:', listResponse.status, tasks);
                
                // Try to update a task (if any exist)
                if (Array.isArray(tasks) && tasks.length > 0) {
                    const taskId = tasks[0].id;
                    console.log(`\nTrying to update task ${taskId}...`);
                    
                    const updateResponse = await fetch(`${apiUrl}/tasks/${taskId}`, {
                        method: 'PUT',
                        headers: {
                            'Authorization': `Bearer ${data.token}`,
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            status: 'in_progress',
                            notes: 'Test update'
                        })
                    });
                    
                    console.log(`Update response status: ${updateResponse.status}`);
                    const updateResult = await updateResponse.text();
                    console.log('Update result:', updateResult);
                } else {
                    // Try creating a task first
                    console.log('\nNo tasks found. Trying to create one...');
                    const createResponse = await fetch(`${apiUrl}/tasks`, {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${data.token}`,
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            title: 'Test Task',
                            description: 'Testing task creation',
                            status: 'pending',
                            priority: 'medium'
                        })
                    });
                    console.log('Create task response:', createResponse.status);
                    const createResult = await createResponse.text();
                    console.log('Create result:', createResult);
                }
                
                break;
            }
        } catch (error) {
            console.log(`Error with ${cred.email}:`, error.message);
        }
    }
    
    // Test 3: Check available endpoints
    console.log('\nChecking common endpoints...');
    const endpoints = ['/api', '/api/tasks', '/api/projects', '/api/users', '/api/auth/status'];
    
    for (const endpoint of endpoints) {
        try {
            const response = await fetch(`http://192.168.1.24:3009${endpoint}`);
            console.log(`${endpoint}: ${response.status} ${response.statusText}`);
        } catch (error) {
            console.log(`${endpoint}: Error - ${error.message}`);
        }
    }
    
    // Test 4: Try update without auth to see error details
    console.log('\nTesting update without auth...');
    try {
        const response = await fetch(`${apiUrl}/tasks/1`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                status: 'in_progress'
            })
        });
        console.log('No auth update status:', response.status);
        const result = await response.text();
        console.log('No auth result:', result);
    } catch (error) {
        console.log('No auth error:', error.message);
    }
}

// Install node-fetch if needed
const { exec } = require('child_process');
exec('npm list node-fetch || npm install node-fetch@2', (error, stdout, stderr) => {
    if (error) {
        console.error('Failed to install node-fetch:', error);
        return;
    }
    
    // Run tests
    testProjectHub().catch(console.error);
});