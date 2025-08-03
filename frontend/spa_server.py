#!/usr/bin/env python3
"""Simple SPA server that serves index.html for all routes"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import sys

class SPAHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory='dist', **kwargs)
    
    def do_GET(self):
        # Serve index.html for all routes (SPA)
        if not os.path.exists(self.translate_path(self.path)):
            self.path = '/index.html'
        return SimpleHTTPRequestHandler.do_GET(self)
    
    def end_headers(self):
        # Add CORS headers for API access
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        SimpleHTTPRequestHandler.end_headers(self)

if __name__ == '__main__':
    port = 3100
    server_address = ('0.0.0.0', port)  # Bind to all interfaces
    
    print(f"🚀 Starting SPA server on http://0.0.0.0:{port}")
    print(f"📡 Accessible at http://192.168.1.25:{port}")
    
    httpd = HTTPServer(server_address, SPAHTTPRequestHandler)
    httpd.serve_forever()