import http.server
import socketserver
import os
import urllib.parse

PORT = 3101
DIRECTORY = "dist"

class SPAHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def do_GET(self):
        # Parse the URL
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        # Check if this is a file request (has an extension)
        if '.' in os.path.basename(path):
            # Serve the file normally
            super().do_GET()
        else:
            # For all other paths, serve index.html (SPA routing)
            self.path = '/index.html'
            super().do_GET()

with socketserver.TCPServer(("", PORT), SPAHTTPRequestHandler) as httpd:
    print(f"Server running at http://localhost:{PORT}/")
    print(f"Serving files from {os.path.abspath(DIRECTORY)}")
    print("SPA routing enabled - all routes will serve index.html")
    httpd.serve_forever()