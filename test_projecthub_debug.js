#!/usr/bin/env node

/**
 * ProjectHub API Debug Script
 * Tests various scenarios to understand the PUT endpoint issue
 */

const fetch = require('node-fetch');

const API_URL = 'http://192.168.1.24:3009/api';

async function testEndpoint(description, method, endpoint, body = null) {
    console.log(`\n=== ${description} ===`);
    
    const options = {
        method: method,
        headers: {
            'Content-Type': 'application/json'
        }
    };
    
    if (body) {
        options.body = JSON.stringify(body);
    }
    
    try {
        const response = await fetch(`${API_URL}${endpoint}`, options);
        const responseText = await response.text();
        
        console.log(`Status: ${response.status} ${response.statusText}`);
        console.log(`Headers:`, Object.fromEntries(response.headers));
        
        try {
            const data = JSON.parse(responseText);
            console.log(`Response:`, JSON.stringify(data, null, 2));
        } catch {
            console.log(`Response (text):`, responseText);
        }
        
        return { status: response.status, data: responseText };
    } catch (error) {
        console.error(`Error:`, error.message);
        return { status: 0, error: error.message };
    }
}

async function runTests() {
    console.log('ProjectHub API Debug Tests');
    console.log('==========================');
    
    // Test 1: Get all tasks
    const tasksResult = await testEndpoint('GET /api/tasks', 'GET', '/tasks');
    
    if (tasksResult.status !== 200) {
        console.error('Failed to get tasks, aborting tests');
        return;
    }
    
    const tasks = JSON.parse(tasksResult.data);
    if (tasks.length === 0) {
        console.log('No tasks found, creating a test task');
        
        // Create a test task
        const createResult = await testEndpoint(
            'POST /api/tasks - Create test task',
            'POST',
            '/tasks',
            {
                project_id: tasks[0]?.project_id || '37e40274-e503-4f66-a9a5-eef8d00c3b88',
                title: 'Test Task for Debugging',
                description: 'This is a test task to debug PUT endpoint',
                status: 'pending',
                priority: 'low'
            }
        );
        
        if (createResult.status === 201 || createResult.status === 200) {
            const newTask = JSON.parse(createResult.data);
            tasks.push(newTask);
        }
    }
    
    // Use the first task for testing
    const testTask = tasks[0];
    console.log(`\nUsing task for tests: ${testTask.title} (ID: ${testTask.id})`);
    
    // Test 2: Try different PUT payloads
    const putTests = [
        {
            desc: 'PUT with only status',
            payload: { status: 'completed' }
        },
        {
            desc: 'PUT with full task object',
            payload: {
                ...testTask,
                status: 'completed'
            }
        },
        {
            desc: 'PUT with minimal fields',
            payload: {
                id: testTask.id,
                status: 'completed'
            }
        },
        {
            desc: 'PUT with project_id and status',
            payload: {
                project_id: testTask.project_id,
                status: 'completed'
            }
        },
        {
            desc: 'PUT with all original fields modified',
            payload: {
                ...testTask,
                status: 'completed',
                updated_at: new Date().toISOString(),
                actual_hours: 2
            }
        }
    ];
    
    for (const test of putTests) {
        await testEndpoint(
            `PUT /api/tasks/${testTask.id} - ${test.desc}`,
            'PUT',
            `/tasks/${testTask.id}`,
            test.payload
        );
    }
    
    // Test 3: Try PATCH method
    await testEndpoint(
        `PATCH /api/tasks/${testTask.id} - Test PATCH method`,
        'PATCH',
        `/tasks/${testTask.id}`,
        { status: 'completed' }
    );
    
    // Test 4: Check if there's a different update endpoint
    const alternativeEndpoints = [
        `/tasks/${testTask.id}/status`,
        `/tasks/${testTask.id}/update`,
        `/task/${testTask.id}`,
        `/tasks/update/${testTask.id}`
    ];
    
    for (const endpoint of alternativeEndpoints) {
        await testEndpoint(
            `PUT ${endpoint} - Alternative endpoint`,
            'PUT',
            endpoint,
            { status: 'completed' }
        );
    }
    
    // Test 5: Check authentication
    await testEndpoint(
        'PUT with Authorization header',
        'PUT',
        `/tasks/${testTask.id}`,
        { status: 'completed' },
        { 'Authorization': 'Bearer test-token' }
    );
}

// Run tests
runTests().catch(console.error);