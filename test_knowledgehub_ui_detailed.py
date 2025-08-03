#!/usr/bin/env python3
"""
Detailed UI testing for KnowledgeHub with better React app handling
"""

import asyncio
import json
import os
from datetime import datetime
from playwright.async_api import async_playwright, Page, expect
from typing import Dict, List

class DetailedKnowledgeHubTester:
    def __init__(self, base_url: str = "http://192.168.1.25:3100"):
        self.base_url = base_url
        self.api_url = "http://192.168.1.25:3000"
        self.screenshots_dir = "ui_test_detailed"
        
    async def setup(self):
        """Initialize browser with debugging capabilities"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=False,  # Run with GUI for better debugging
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        self.context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080}
        )
        self.page = await self.context.new_page()
        
        # Enable console logging
        self.page.on("console", lambda msg: print(f"CONSOLE {msg.type}: {msg.text}"))
        self.page.on("pageerror", lambda err: print(f"PAGE ERROR: {err}"))
        
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
    async def wait_for_react_app(self):
        """Wait for React app to properly load"""
        print("Waiting for React app to initialize...")
        
        try:
            # Wait for the root element to have content
            await self.page.wait_for_function(
                "document.querySelector('#root') && document.querySelector('#root').children.length > 0",
                timeout=30000
            )
            
            # Additional wait for any loading states to complete
            await self.page.wait_for_timeout(2000)
            
            print("React app loaded successfully")
            return True
        except Exception as e:
            print(f"Failed to load React app: {e}")
            return False
            
    async def setup_authentication(self):
        """Set up localStorage authentication before app loads"""
        print("Setting up authentication...")
        
        # Navigate to the base URL first
        await self.page.goto(self.base_url)
        
        # Set authentication in localStorage
        auth_data = {
            "apiUrl": self.api_url,
            "apiKey": "knhub_V05H0-fZ_kUJB93um_pS00Nxv3i60gZogNGhMLtTFbM",
            "enableNotifications": True,
            "autoRefresh": True,
            "refreshInterval": 30,
            "darkMode": False,
            "language": "en",
            "animationSpeed": 1,
            "cacheSize": 100,
            "maxMemories": 1000,
            "compressionEnabled": True
        }
        
        await self.page.evaluate(f"""
            localStorage.setItem('knowledgehub_settings', '{json.dumps(auth_data)}');
            console.log('Authentication set in localStorage');
        """)
        
        # Reload to apply settings
        await self.page.reload()
        
        # Wait for app to load with authentication
        await self.wait_for_react_app()
        
    async def take_screenshot(self, name: str):
        """Take a screenshot with detailed info"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.screenshots_dir}/{timestamp}_{name}.png"
        await self.page.screenshot(path=filename, full_page=True)
        print(f"Screenshot saved: {filename}")
        return filename
        
    async def analyze_page_structure(self):
        """Analyze the current page structure for debugging"""
        print("\nAnalyzing page structure...")
        
        # Get page title
        title = await self.page.title()
        print(f"Page title: {title}")
        
        # Get all visible text
        visible_text = await self.page.evaluate("""
            () => {
                const walker = document.createTreeWalker(
                    document.body,
                    NodeFilter.SHOW_TEXT,
                    null,
                    false
                );
                const texts = [];
                let node;
                while (node = walker.nextNode()) {
                    const text = node.textContent.trim();
                    if (text.length > 0) {
                        texts.push(text);
                    }
                }
                return texts.slice(0, 20); // First 20 text nodes
            }
        """)
        print(f"Visible text (first 20 items): {visible_text}")
        
        # Get all clickable elements
        clickable = await self.page.evaluate("""
            () => {
                const elements = document.querySelectorAll('a, button, [role="button"], [onclick]');
                return Array.from(elements).map(el => ({
                    tag: el.tagName,
                    text: el.textContent.trim().slice(0, 50),
                    href: el.href || 'N/A'
                })).slice(0, 10);
            }
        """)
        print(f"Clickable elements (first 10): {json.dumps(clickable, indent=2)}")
        
        # Get all form inputs
        inputs = await self.page.evaluate("""
            () => {
                const elements = document.querySelectorAll('input, textarea, select');
                return Array.from(elements).map(el => ({
                    tag: el.tagName,
                    type: el.type || 'N/A',
                    name: el.name || 'N/A',
                    placeholder: el.placeholder || 'N/A'
                }));
            }
        """)
        print(f"Form inputs: {json.dumps(inputs, indent=2)}")
        
        # Check for common React/Material-UI elements
        mui_elements = await self.page.evaluate("""
            () => {
                const muiClasses = ['MuiButton', 'MuiCard', 'MuiTextField', 'MuiAppBar', 'MuiDrawer'];
                const found = {};
                muiClasses.forEach(cls => {
                    const elements = document.querySelectorAll(`[class*="${cls}"]`);
                    if (elements.length > 0) {
                        found[cls] = elements.length;
                    }
                });
                return found;
            }
        """)
        print(f"Material-UI components found: {json.dumps(mui_elements, indent=2)}")
        
    async def test_navigation(self):
        """Test navigation between pages"""
        print("\n=== Testing Navigation ===")
        
        pages = [
            ("/", "Home"),
            ("/ai-intelligence", "AI Intelligence"),
            ("/memory", "Memory System"),
            ("/search", "Search"),
            ("/sources", "Sources"),
            ("/settings", "Settings")
        ]
        
        results = []
        
        for path, name in pages:
            print(f"\nNavigating to {name} ({path})...")
            
            try:
                # Navigate to page
                await self.page.goto(f"{self.base_url}{path}")
                
                # Wait for React to render
                await self.wait_for_react_app()
                
                # Take screenshot
                screenshot = await self.take_screenshot(f"{name.lower().replace(' ', '_')}")
                
                # Analyze page
                await self.analyze_page_structure()
                
                # Check if we're on the right page by looking for route-specific content
                current_url = self.page.url
                print(f"Current URL: {current_url}")
                
                results.append({
                    "page": name,
                    "path": path,
                    "status": "success",
                    "screenshot": screenshot,
                    "url": current_url
                })
                
            except Exception as e:
                print(f"ERROR navigating to {name}: {e}")
                results.append({
                    "page": name,
                    "path": path,
                    "status": "failed",
                    "error": str(e)
                })
                
        return results
        
    async def test_api_connectivity(self):
        """Test if the UI can connect to the API"""
        print("\n=== Testing API Connectivity ===")
        
        # Go to a page that makes API calls
        await self.page.goto(f"{self.base_url}/memory")
        await self.wait_for_react_app()
        
        # Monitor network requests
        api_calls = []
        
        async def log_request(request):
            if self.api_url in request.url:
                api_calls.append({
                    "url": request.url,
                    "method": request.method,
                    "headers": await request.all_headers()
                })
                
        self.page.on("request", log_request)
        
        # Wait for API calls
        await self.page.wait_for_timeout(5000)
        
        print(f"API calls made: {len(api_calls)}")
        for call in api_calls[:5]:  # Show first 5
            print(f"  {call['method']} {call['url']}")
            
        return api_calls
        
    async def test_interactive_features(self):
        """Test interactive features on each page"""
        print("\n=== Testing Interactive Features ===")
        
        # Test search functionality
        await self.page.goto(f"{self.base_url}/search")
        await self.wait_for_react_app()
        
        # Find search input
        search_inputs = await self.page.query_selector_all('input')
        print(f"Found {len(search_inputs)} input fields")
        
        if search_inputs:
            # Try the first input
            await search_inputs[0].fill("test query")
            await self.take_screenshot("search_with_query")
            
            # Look for search button
            buttons = await self.page.query_selector_all('button')
            print(f"Found {len(buttons)} buttons")
            
            for button in buttons:
                text = await button.text_content()
                print(f"  Button: {text}")
                
    async def run_all_tests(self):
        """Run all detailed tests"""
        print("🚀 Starting Detailed KnowledgeHub UI Testing")
        print(f"Target: {self.base_url}")
        print("-" * 60)
        
        await self.setup()
        await self.setup_authentication()
        
        # Run tests
        navigation_results = await self.test_navigation()
        api_results = await self.test_api_connectivity()
        await self.test_interactive_features()
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
            "navigation_results": navigation_results,
            "api_calls_detected": len(api_results) if isinstance(api_results, list) else 0,
            "screenshots_dir": self.screenshots_dir
        }
        
        report_file = f"detailed_ui_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📄 Report saved to: {report_file}")
        
        # Keep browser open for manual inspection
        print("\n⚠️  Browser will stay open for 30 seconds for manual inspection...")
        await self.page.wait_for_timeout(30000)
        
        await self.browser.close()
        await self.playwright.stop()

async def main():
    tester = DetailedKnowledgeHubTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())