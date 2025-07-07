#!/usr/bin/env node

/**
 * Final comprehensive test to verify ProjectHub PUT endpoint status
 */

const fetch = require('node-fetch');

const PROXY_URL = 'http://localhost:3109/api';
const ORIGINAL_URL = 'http://192.168.1.24:3009/api';

async function testPutEndpoint(description, url, taskId, updateData) {
    console.log(`\n🧪 ${description}`);
    console.log(`   URL: PUT ${url}/tasks/${taskId}`);
    console.log(`   Data: ${JSON.stringify(updateData)}`);
    
    try {
        const response = await fetch(`${url}/tasks/${taskId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(updateData)
        });
        
        const responseText = await response.text();
        
        console.log(`   Status: ${response.status} ${response.statusText}`);
        
        if (response.ok) {
            console.log(`   ✅ Success`);
            
            try {
                const data = JSON.parse(responseText);
                console.log(`   📝 Updated status: ${data.status}`);
                if (data.update_method) {
                    console.log(`   🔧 Update method: ${data.update_method}`);
                }
                return true;
            } catch (e) {
                console.log(`   ⚠️  Response not JSON: ${responseText}`);
                return false;
            }
        } else {
            console.log(`   ❌ Failed: ${responseText}`);
            return false;
        }
    } catch (error) {
        console.log(`   ❌ Error: ${error.message}`);
        return false;
    }
}

async function runFinalTest() {
    console.log('Final ProjectHub PUT Endpoint Test');
    console.log('==================================\n');
    
    // Get a task to test with
    console.log('Getting tasks for testing...');
    const tasksResponse = await fetch(`${ORIGINAL_URL}/tasks`);
    const tasks = await tasksResponse.json();
    
    // Find a completed task that we can toggle
    const testTask = tasks.find(t => t.status === 'completed' && t.title.includes('Test Task'));
    
    if (!testTask) {
        console.log('⚠️  No suitable test task found. Creating one...');
        
        // Create a test task
        const createResponse = await fetch(`${ORIGINAL_URL}/tasks`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                project_id: tasks[0].project_id,
                title: 'PUT Endpoint Test Task',
                description: 'Task created to test PUT endpoint functionality',
                status: 'pending',
                priority: 'low'
            })
        });
        
        if (createResponse.ok) {
            const newTask = await createResponse.json();
            console.log(`✅ Created test task: ${newTask.id}`);
            
            // Test both endpoints with the new task
            await testBothEndpoints(newTask.id);
        } else {
            console.log('❌ Failed to create test task');
        }
    } else {
        console.log(`📋 Using existing test task: ${testTask.title} (${testTask.id})`);
        await testBothEndpoints(testTask.id);
    }
}

async function testBothEndpoints(taskId) {
    console.log('\n--- Testing Both Endpoints ---');
    
    // Test 1: Original API - set to in_progress
    const originalTest1 = await testPutEndpoint(
        'Original API: Set to in_progress',
        ORIGINAL_URL,
        taskId,
        { status: 'in_progress' }
    );
    
    // Test 2: Proxy API - set to completed
    const proxyTest1 = await testPutEndpoint(
        'Proxy API: Set to completed',
        PROXY_URL,
        taskId,
        { status: 'completed' }
    );
    
    // Test 3: Original API - set to pending
    const originalTest2 = await testPutEndpoint(
        'Original API: Set to pending',
        ORIGINAL_URL,
        taskId,
        { status: 'pending' }
    );
    
    // Test 4: Proxy API - set to completed again
    const proxyTest2 = await testPutEndpoint(
        'Proxy API: Set to completed again',
        PROXY_URL,
        taskId,
        { status: 'completed' }
    );
    
    // Summary
    console.log('\n🎯 Final Test Results');
    console.log('====================');
    console.log(`Original API Test 1 (in_progress): ${originalTest1 ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`Proxy API Test 1 (completed): ${proxyTest1 ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`Original API Test 2 (pending): ${originalTest2 ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`Proxy API Test 2 (completed): ${proxyTest2 ? '✅ PASS' : '❌ FAIL'}`);
    
    const originalWorking = originalTest1 && originalTest2;
    const proxyWorking = proxyTest1 && proxyTest2;
    
    console.log('\n📊 Endpoint Status');
    console.log('==================');
    console.log(`Original API (http://192.168.1.24:3009/api): ${originalWorking ? '✅ WORKING' : '❌ BROKEN'}`);
    console.log(`Proxy API (http://localhost:3109/api): ${proxyWorking ? '✅ WORKING' : '❌ BROKEN'}`);
    
    console.log('\n💡 Recommendation');
    console.log('=================');
    if (originalWorking) {
        console.log('✅ Original API is now working! The PUT endpoint has been fixed.');
        console.log('   You can use either endpoint:');
        console.log('   - Original: http://192.168.1.24:3009/api (direct)');
        console.log('   - Proxy: http://localhost:3109/api (with additional features)');
    } else {
        console.log('❌ Original API still has issues. Use the proxy:');
        console.log('   - Proxy: http://localhost:3109/api (recommended)');
    }
    
    if (proxyWorking) {
        console.log('✅ Proxy is working correctly and provides additional features.');
    } else {
        console.log('❌ Proxy has issues. Please check the proxy server.');
    }
}

// Run the final test
runFinalTest().catch(console.error);