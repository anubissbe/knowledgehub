const http = require('http');

// Test CORS headers directly
const options = {
  hostname: '192.168.1.25',
  port: 3000,
  path: '/api/ai-features/summary',
  method: 'OPTIONS',
  headers: {
    'Origin': 'http://192.168.1.25:3100',
    'Access-Control-Request-Method': 'GET',
    'Access-Control-Request-Headers': 'X-API-Key',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
  }
};

console.log('Testing CORS preflight request...\n');
console.log('Request details:');
console.log(`URL: http://${options.hostname}:${options.port}${options.path}`);
console.log(`Method: ${options.method}`);
console.log('Headers:', options.headers);
console.log('\n---\n');

const req = http.request(options, (res) => {
  console.log(`Status Code: ${res.statusCode}`);
  console.log('\nResponse Headers:');
  
  // Look for CORS headers
  const corsHeaders = [
    'access-control-allow-origin',
    'access-control-allow-methods',
    'access-control-allow-headers',
    'access-control-allow-credentials',
    'access-control-max-age',
    'vary'
  ];
  
  corsHeaders.forEach(header => {
    if (res.headers[header]) {
      console.log(`${header}: ${res.headers[header]}`);
    }
  });
  
  // Also check for any other access-control headers
  Object.keys(res.headers).forEach(header => {
    if (header.startsWith('access-control-') && !corsHeaders.includes(header)) {
      console.log(`${header}: ${res.headers[header]}`);
    }
  });
  
  let data = '';
  res.on('data', chunk => data += chunk);
  res.on('end', () => {
    if (data) {
      console.log('\nResponse body:', data);
    }
  });
});

req.on('error', (e) => {
  console.error(`Request error: ${e.message}`);
});

req.end();