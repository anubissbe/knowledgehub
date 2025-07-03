// Simple test to check if the frontend is working
import axios from 'axios';

async function testFrontend() {
  try {
    // Test if the frontend is serving
    const response = await axios.get('http://localhost:3001/');
    console.log('Frontend is serving HTML:', response.data.includes('<!doctype html>'));
    
    // Test if vite client is loading
    const viteResponse = await axios.get('http://localhost:3001/@vite/client');
    console.log('Vite client is available:', viteResponse.status === 200);
    
    // Test if main.tsx is being served
    const mainResponse = await axios.get('http://localhost:3001/src/main.tsx');
    console.log('Main.tsx is being served:', mainResponse.status === 200);
    
  } catch (error) {
    console.error('Error testing frontend:', error.message);
  }
}

testFrontend();