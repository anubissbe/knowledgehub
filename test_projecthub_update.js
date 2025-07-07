const fetch = require('node-fetch');

async function testProjectHubUpdate() {
    const apiUrl = 'http://192.168.1.24:3009/api';
    
    console.log('Testing ProjectHub Update Functionality...\n');
    
    // Try to login with correct credentials
    const credentials = [
        { email: 'claude@projecthub.com', password: 'admin123' },
        { email: 'claude@projecthub.com', password: 'claude123' },
        { email: 'claude@projecthub.com', password: 'password' },
        { email: 'bert@telkom.be', password: 'admin123' },
        { email: 'bert@telkom.be', password: 'bert123' },
        { email: 'bert@telkom.be', password: 'password' }
    ];
    
    let token = null;
    
    for (const cred of credentials) {
        try {
            console.log(`Trying to login with ${cred.email} / ${cred.password}...`);
            const response = await fetch(`${apiUrl}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(cred)
            });
            const data = await response.json();
            
            if (response.ok && data.token) {
                console.log('✓ Authentication successful!');
                token = data.token;
                break;
            } else {
                console.log(`✗ Failed: ${data.error || response.statusText}`);
            }
        } catch (error) {
            console.log(`✗ Error: ${error.message}`);
        }
    }
    
    if (!token) {
        console.log('\nCould not authenticate. Proceeding with unauthenticated tests...');
    }
    
    // Get a task to update
    console.log('\nFetching tasks...');
    const tasksResponse = await fetch(`${apiUrl}/tasks`);
    const tasks = await tasksResponse.json();
    
    if (!Array.isArray(tasks) || tasks.length === 0) {
        console.log('No tasks found to test update.');
        return;
    }
    
    const testTask = tasks[0];
    console.log(`\nTesting update on task: ${testTask.id} - "${testTask.title}"`);
    console.log(`Current status: ${testTask.status}`);
    
    // Test different update scenarios
    const updateTests = [
        {
            name: 'Update with authentication',
            headers: token ? {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            } : {
                'Content-Type': 'application/json'
            },
            body: {
                status: 'in_progress',
                notes: 'Testing update functionality'
            }
        },
        {
            name: 'Update without authentication',
            headers: {
                'Content-Type': 'application/json'
            },
            body: {
                status: 'in_progress'
            }
        },
        {
            name: 'Update with minimal data',
            headers: token ? {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            } : {
                'Content-Type': 'application/json'
            },
            body: {
                notes: 'Just updating notes'
            }
        },
        {
            name: 'Update with all fields',
            headers: token ? {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            } : {
                'Content-Type': 'application/json'
            },
            body: {
                title: testTask.title,
                description: testTask.description,
                status: 'in_progress',
                priority: testTask.priority,
                notes: 'Full update test'
            }
        }
    ];
    
    for (const test of updateTests) {
        console.log(`\n${test.name}:`);
        try {
            const response = await fetch(`${apiUrl}/tasks/${testTask.id}`, {
                method: 'PUT',
                headers: test.headers,
                body: JSON.stringify(test.body)
            });
            
            console.log(`Response status: ${response.status} ${response.statusText}`);
            
            const responseText = await response.text();
            try {
                const responseData = JSON.parse(responseText);
                console.log('Response:', JSON.stringify(responseData, null, 2));
            } catch {
                console.log('Response text:', responseText);
            }
            
            // If we get a 500 error, check server headers for clues
            if (response.status === 500) {
                console.log('Response headers:');
                for (const [key, value] of response.headers.entries()) {
                    console.log(`  ${key}: ${value}`);
                }
            }
        } catch (error) {
            console.log(`Error: ${error.message}`);
            console.log('Stack:', error.stack);
        }
    }
    
    // Try to understand the API structure better
    console.log('\n\nChecking API structure...');
    
    // Test OPTIONS request to see allowed methods
    try {
        const optionsResponse = await fetch(`${apiUrl}/tasks/${testTask.id}`, {
            method: 'OPTIONS'
        });
        console.log(`OPTIONS response: ${optionsResponse.status}`);
        console.log('Allow header:', optionsResponse.headers.get('allow'));
        console.log('Access-Control-Allow-Methods:', optionsResponse.headers.get('access-control-allow-methods'));
    } catch (error) {
        console.log('OPTIONS request failed:', error.message);
    }
    
    // Try PATCH instead of PUT
    console.log('\nTrying PATCH instead of PUT...');
    try {
        const patchResponse = await fetch(`${apiUrl}/tasks/${testTask.id}`, {
            method: 'PATCH',
            headers: token ? {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            } : {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                status: 'in_progress'
            })
        });
        console.log(`PATCH response: ${patchResponse.status} ${patchResponse.statusText}`);
        const patchResult = await patchResponse.text();
        console.log('PATCH result:', patchResult);
    } catch (error) {
        console.log('PATCH request failed:', error.message);
    }
}

// Run the test
testProjectHubUpdate().catch(console.error);