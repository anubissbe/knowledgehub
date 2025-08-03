#!/usr/bin/env python3
"""
Start multiple scraper workers to process documentation sources
"""

import subprocess
import os
import time
import signal
import sys

# Number of workers to start
NUM_WORKERS = 3

workers = []

def signal_handler(sig, frame):
    print("\nShutting down scraper workers...")
    for proc in workers:
        proc.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

print(f"Starting {NUM_WORKERS} scraper workers...")

# Start workers
for i in range(NUM_WORKERS):
    print(f"Starting worker {i+1}...")
    proc = subprocess.Popen([
        sys.executable, 
        "scraper_worker.py"
    ], 
    stdout=open(f"scraper_worker_{i+1}.log", "w"),
    stderr=subprocess.STDOUT
    )
    workers.append(proc)
    time.sleep(1)  # Stagger startup

print(f"\n✅ Started {NUM_WORKERS} scraper workers")
print("Press Ctrl+C to stop all workers\n")

# Monitor workers
try:
    while True:
        # Check if any workers have crashed
        for i, proc in enumerate(workers):
            if proc.poll() is not None:
                print(f"Worker {i+1} crashed with code {proc.returncode}, restarting...")
                proc = subprocess.Popen([
                    sys.executable, 
                    "scraper_worker.py"
                ], 
                stdout=open(f"scraper_worker_{i+1}.log", "a"),
                stderr=subprocess.STDOUT
                )
                workers[i] = proc
        
        time.sleep(5)  # Check every 5 seconds
        
except KeyboardInterrupt:
    signal_handler(None, None)