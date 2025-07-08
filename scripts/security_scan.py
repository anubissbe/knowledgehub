#!/usr/bin/env python3
"""
KnowledgeHub Security Dependency Scanner

Comprehensive security scanning for Python dependencies, Docker images,
and JavaScript packages with automated reporting and alerting.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import requests
import yaml
from collections import defaultdict

class SecurityScanner:
    """Main security scanner for dependencies"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.scan_results = {
            "scan_date": datetime.utcnow().isoformat(),
            "project": "KnowledgeHub",
            "vulnerabilities": {
                "critical": [],
                "high": [],
                "medium": [],
                "low": []
            },
            "summary": {},
            "recommendations": []
        }
        
    def run_full_scan(self) -> Dict[str, Any]:
        """Run comprehensive security scan"""
        print("🔒 KnowledgeHub Security Dependency Scanner")
        print("=" * 50)
        
        # Python dependency scan
        print("\n📦 Scanning Python dependencies...")
        self.scan_python_dependencies()
        
        # JavaScript dependency scan
        print("\n📦 Scanning JavaScript dependencies...")
        self.scan_javascript_dependencies()
        
        # Docker image scan
        print("\n🐳 Scanning Docker images...")
        self.scan_docker_images()
        
        # License compliance check
        print("\n📜 Checking license compliance...")
        self.check_license_compliance()
        
        # Generate recommendations
        self.generate_recommendations()
        
        # Generate report
        self.generate_report()
        
        return self.scan_results
    
    def scan_python_dependencies(self):
        """Scan Python dependencies for vulnerabilities"""
        scanners = [
            self._scan_with_safety,
            self._scan_with_pip_audit,
            self._scan_with_bandit
        ]
        
        for scanner in scanners:
            try:
                scanner()
            except Exception as e:
                print(f"  ⚠️ {scanner.__name__} failed: {e}")
    
    def _scan_with_safety(self):
        """Scan with safety package"""
        try:
            # Check if safety is installed
            subprocess.run(["safety", "--version"], capture_output=True, check=True)
            
            # Run safety check
            result = subprocess.run(
                ["safety", "check", "--json", "--full-report"],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                vulnerabilities = json.loads(result.stdout)
                for vuln in vulnerabilities:
                    self._add_vulnerability({
                        "type": "python",
                        "package": vuln.get("package"),
                        "installed_version": vuln.get("installed_version"),
                        "vulnerability": vuln.get("vulnerability"),
                        "severity": self._map_safety_severity(vuln.get("severity", "unknown")),
                        "description": vuln.get("description"),
                        "fixed_in": vuln.get("fixed_in", []),
                        "cve": vuln.get("cve"),
                        "scanner": "safety"
                    })
            
            print("  ✅ Safety scan completed")
            
        except subprocess.CalledProcessError:
            print("  ⚠️ Safety not installed, skipping")
        except Exception as e:
            print(f"  ❌ Safety scan error: {e}")
    
    def _scan_with_pip_audit(self):
        """Scan with pip-audit"""
        try:
            # Check if pip-audit is installed
            subprocess.run(["pip-audit", "--version"], capture_output=True, check=True)
            
            # Run pip-audit
            result = subprocess.run(
                ["pip-audit", "--format", "json", "--desc"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.stdout:
                audit_results = json.loads(result.stdout)
                for vuln in audit_results.get("vulnerabilities", []):
                    self._add_vulnerability({
                        "type": "python",
                        "package": vuln.get("name"),
                        "installed_version": vuln.get("version"),
                        "vulnerability": vuln.get("id"),
                        "severity": self._normalize_severity(vuln.get("fix_versions", [])),
                        "description": vuln.get("description"),
                        "fixed_in": vuln.get("fix_versions", []),
                        "aliases": vuln.get("aliases", []),
                        "scanner": "pip-audit"
                    })
            
            print("  ✅ Pip-audit scan completed")
            
        except subprocess.CalledProcessError:
            print("  ⚠️ Pip-audit not installed, skipping")
        except Exception as e:
            print(f"  ❌ Pip-audit scan error: {e}")
    
    def _scan_with_bandit(self):
        """Scan Python code with bandit for security issues"""
        try:
            # Check if bandit is installed
            subprocess.run(["bandit", "--version"], capture_output=True, check=True)
            
            # Run bandit
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json", "-ll"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.stdout:
                bandit_results = json.loads(result.stdout)
                for issue in bandit_results.get("results", []):
                    self._add_vulnerability({
                        "type": "code",
                        "file": issue.get("filename"),
                        "line": issue.get("line_number"),
                        "vulnerability": issue.get("test_name"),
                        "severity": issue.get("issue_severity", "MEDIUM").lower(),
                        "description": issue.get("issue_text"),
                        "confidence": issue.get("issue_confidence"),
                        "cwe": issue.get("issue_cwe", {}).get("id"),
                        "scanner": "bandit"
                    })
            
            print("  ✅ Bandit scan completed")
            
        except subprocess.CalledProcessError:
            print("  ⚠️ Bandit not installed, skipping")
        except Exception as e:
            print(f"  ❌ Bandit scan error: {e}")
    
    def scan_javascript_dependencies(self):
        """Scan JavaScript dependencies for vulnerabilities"""
        # Find package.json files
        package_files = list(self.project_root.rglob("package.json"))
        
        for package_file in package_files:
            if "node_modules" in str(package_file):
                continue
                
            print(f"  📄 Scanning {package_file.relative_to(self.project_root)}")
            
            # NPM audit
            self._run_npm_audit(package_file.parent)
            
            # Yarn audit if yarn.lock exists
            if (package_file.parent / "yarn.lock").exists():
                self._run_yarn_audit(package_file.parent)
    
    def _run_npm_audit(self, directory: Path):
        """Run npm audit"""
        try:
            result = subprocess.run(
                ["npm", "audit", "--json"],
                capture_output=True,
                text=True,
                cwd=directory
            )
            
            if result.stdout:
                audit_data = json.loads(result.stdout)
                
                for advisory_id, advisory in audit_data.get("vulnerabilities", {}).items():
                    self._add_vulnerability({
                        "type": "javascript",
                        "package": advisory.get("name"),
                        "severity": advisory.get("severity", "unknown"),
                        "vulnerability": f"npm-{advisory_id}",
                        "description": advisory.get("title", ""),
                        "fixed_in": advisory.get("fixAvailable", {}).get("name", ""),
                        "cwe": advisory.get("cwe", []),
                        "url": advisory.get("url", ""),
                        "scanner": "npm-audit"
                    })
            
            print("    ✅ NPM audit completed")
            
        except subprocess.CalledProcessError as e:
            print(f"    ⚠️ NPM audit failed: {e}")
        except Exception as e:
            print(f"    ❌ NPM audit error: {e}")
    
    def _run_yarn_audit(self, directory: Path):
        """Run yarn audit"""
        try:
            result = subprocess.run(
                ["yarn", "audit", "--json"],
                capture_output=True,
                text=True,
                cwd=directory
            )
            
            # Parse yarn audit output (NDJSON format)
            for line in result.stdout.strip().split('\n'):
                if line:
                    try:
                        data = json.loads(line)
                        if data.get("type") == "auditAdvisory":
                            advisory = data.get("data", {}).get("advisory", {})
                            self._add_vulnerability({
                                "type": "javascript",
                                "package": advisory.get("module_name"),
                                "severity": advisory.get("severity", "unknown"),
                                "vulnerability": f"yarn-{advisory.get('id')}",
                                "description": advisory.get("title", ""),
                                "patched_versions": advisory.get("patched_versions", ""),
                                "cwe": advisory.get("cwe", ""),
                                "url": advisory.get("url", ""),
                                "scanner": "yarn-audit"
                            })
                    except json.JSONDecodeError:
                        continue
            
            print("    ✅ Yarn audit completed")
            
        except subprocess.CalledProcessError as e:
            print(f"    ⚠️ Yarn audit failed: {e}")
        except Exception as e:
            print(f"    ❌ Yarn audit error: {e}")
    
    def scan_docker_images(self):
        """Scan Docker images for vulnerabilities"""
        # Find Dockerfiles
        dockerfiles = list(self.project_root.rglob("Dockerfile*"))
        
        for dockerfile in dockerfiles:
            if "node_modules" in str(dockerfile):
                continue
                
            print(f"  📄 Scanning {dockerfile.relative_to(self.project_root)}")
            
            # Extract base images
            base_images = self._extract_base_images(dockerfile)
            
            for image in base_images:
                self._scan_docker_image(image)
    
    def _extract_base_images(self, dockerfile: Path) -> List[str]:
        """Extract base images from Dockerfile"""
        images = []
        
        try:
            with open(dockerfile, 'r') as f:
                for line in f:
                    if line.strip().startswith("FROM "):
                        image = line.strip().split(" ")[1].split(" ")[0]
                        images.append(image)
        except Exception as e:
            print(f"    ❌ Error reading Dockerfile: {e}")
        
        return images
    
    def _scan_docker_image(self, image: str):
        """Scan Docker image with trivy"""
        try:
            # Check if trivy is installed
            subprocess.run(["trivy", "--version"], capture_output=True, check=True)
            
            # Run trivy scan
            result = subprocess.run(
                ["trivy", "image", "--format", "json", "--quiet", image],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                scan_results = json.loads(result.stdout)
                
                for target in scan_results.get("Results", []):
                    for vuln in target.get("Vulnerabilities", []):
                        self._add_vulnerability({
                            "type": "docker",
                            "image": image,
                            "package": vuln.get("PkgName"),
                            "installed_version": vuln.get("InstalledVersion"),
                            "vulnerability": vuln.get("VulnerabilityID"),
                            "severity": vuln.get("Severity", "UNKNOWN").lower(),
                            "description": vuln.get("Title", ""),
                            "fixed_in": vuln.get("FixedVersion", ""),
                            "cve": vuln.get("VulnerabilityID", ""),
                            "scanner": "trivy"
                        })
            
            print(f"    ✅ Trivy scan completed for {image}")
            
        except subprocess.CalledProcessError:
            print("    ⚠️ Trivy not installed, skipping Docker scan")
        except Exception as e:
            print(f"    ❌ Trivy scan error: {e}")
    
    def check_license_compliance(self):
        """Check license compliance for dependencies"""
        # Python licenses
        self._check_python_licenses()
        
        # JavaScript licenses
        self._check_javascript_licenses()
    
    def _check_python_licenses(self):
        """Check Python package licenses"""
        try:
            result = subprocess.run(
                ["pip-licenses", "--format", "json"],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                licenses = json.loads(result.stdout)
                
                # Track problematic licenses
                problematic_licenses = ["GPL", "AGPL", "LGPL"]
                
                for package in licenses:
                    license_name = package.get("License", "Unknown")
                    
                    for prob_license in problematic_licenses:
                        if prob_license in license_name:
                            self.scan_results["recommendations"].append({
                                "type": "license",
                                "package": package.get("Name"),
                                "license": license_name,
                                "recommendation": f"Review {prob_license} license compatibility"
                            })
            
            print("  ✅ Python license check completed")
            
        except subprocess.CalledProcessError:
            print("  ⚠️ pip-licenses not installed, skipping")
        except Exception as e:
            print(f"  ❌ License check error: {e}")
    
    def _check_javascript_licenses(self):
        """Check JavaScript package licenses"""
        # This would use license-checker or similar tool
        pass
    
    def _add_vulnerability(self, vuln: Dict[str, Any]):
        """Add vulnerability to results"""
        severity = vuln.get("severity", "unknown").lower()
        
        if severity in ["critical", "high", "medium", "low"]:
            self.scan_results["vulnerabilities"][severity].append(vuln)
    
    def _map_safety_severity(self, safety_severity: str) -> str:
        """Map safety severity to standard levels"""
        mapping = {
            "high": "high",
            "medium": "medium",
            "low": "low",
            "unknown": "low"
        }
        return mapping.get(safety_severity.lower(), "low")
    
    def _normalize_severity(self, fix_versions: List[str]) -> str:
        """Normalize severity based on fix availability"""
        if not fix_versions:
            return "critical"
        return "high"
    
    def generate_recommendations(self):
        """Generate security recommendations based on scan results"""
        vulns = self.scan_results["vulnerabilities"]
        
        # Critical vulnerabilities
        if vulns["critical"]:
            self.scan_results["recommendations"].append({
                "priority": "CRITICAL",
                "title": "Address Critical Vulnerabilities Immediately",
                "description": f"Found {len(vulns['critical'])} critical vulnerabilities that require immediate attention",
                "action": "Update affected packages to patched versions or apply workarounds"
            })
        
        # High vulnerabilities
        if vulns["high"]:
            self.scan_results["recommendations"].append({
                "priority": "HIGH",
                "title": "Fix High Severity Vulnerabilities",
                "description": f"Found {len(vulns['high'])} high severity vulnerabilities",
                "action": "Plan updates for next release cycle"
            })
        
        # Outdated dependencies
        self._check_outdated_dependencies()
        
        # Security headers
        self.scan_results["recommendations"].append({
            "priority": "MEDIUM",
            "title": "Regular Security Scanning",
            "description": "Implement automated security scanning in CI/CD pipeline",
            "action": "Add security scanning to GitHub Actions workflow"
        })
    
    def _check_outdated_dependencies(self):
        """Check for outdated dependencies"""
        # This would check for significantly outdated packages
        pass
    
    def generate_report(self):
        """Generate security scan report"""
        # Calculate summary
        total_vulns = sum(
            len(vulns) for vulns in self.scan_results["vulnerabilities"].values()
        )
        
        self.scan_results["summary"] = {
            "total_vulnerabilities": total_vulns,
            "critical": len(self.scan_results["vulnerabilities"]["critical"]),
            "high": len(self.scan_results["vulnerabilities"]["high"]),
            "medium": len(self.scan_results["vulnerabilities"]["medium"]),
            "low": len(self.scan_results["vulnerabilities"]["low"]),
            "scan_duration": "0:00:00",  # Would calculate actual duration
            "scanners_used": ["safety", "pip-audit", "bandit", "npm-audit", "trivy"]
        }
        
        # Save JSON report
        report_path = self.project_root / "security-scan-report.json"
        with open(report_path, 'w') as f:
            json.dump(self.scan_results, f, indent=2)
        
        # Save HTML report
        self._generate_html_report()
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 SCAN SUMMARY")
        print("=" * 50)
        print(f"Total vulnerabilities: {total_vulns}")
        print(f"  🔴 Critical: {self.scan_results['summary']['critical']}")
        print(f"  🟠 High: {self.scan_results['summary']['high']}")
        print(f"  🟡 Medium: {self.scan_results['summary']['medium']}")
        print(f"  🟢 Low: {self.scan_results['summary']['low']}")
        print(f"\n📄 Report saved to: {report_path}")
    
    def _generate_html_report(self):
        """Generate HTML report"""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>KnowledgeHub Security Scan Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .critical {{ color: #d32f2f; }}
        .high {{ color: #f57c00; }}
        .medium {{ color: #fbc02d; }}
        .low {{ color: #388e3c; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        .summary {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .recommendation {{ background-color: #fff3e0; padding: 10px; margin: 10px 0; border-left: 4px solid #ff9800; }}
    </style>
</head>
<body>
    <h1>KnowledgeHub Security Scan Report</h1>
    <p>Scan Date: {scan_date}</p>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Total Vulnerabilities: <strong>{total}</strong></p>
        <ul>
            <li class="critical">Critical: {critical}</li>
            <li class="high">High: {high}</li>
            <li class="medium">Medium: {medium}</li>
            <li class="low">Low: {low}</li>
        </ul>
    </div>
    
    <h2>Recommendations</h2>
    {recommendations}
    
    <h2>Vulnerability Details</h2>
    {vulnerability_tables}
    
    <footer>
        <p>Generated by KnowledgeHub Security Scanner</p>
    </footer>
</body>
</html>
        """
        
        # Build recommendations HTML
        recommendations_html = ""
        for rec in self.scan_results["recommendations"]:
            priority = rec.get("priority", "MEDIUM")
            recommendations_html += f"""
            <div class="recommendation">
                <strong>[{priority}]</strong> {rec.get('title', '')}<br>
                {rec.get('description', '')}<br>
                <em>Action: {rec.get('action', '')}</em>
            </div>
            """
        
        # Build vulnerability tables
        vuln_tables_html = ""
        for severity, vulns in self.scan_results["vulnerabilities"].items():
            if vulns:
                vuln_tables_html += f"<h3 class='{severity}'>{severity.upper()} Vulnerabilities</h3>"
                vuln_tables_html += "<table><tr><th>Type</th><th>Package</th><th>Vulnerability</th><th>Description</th><th>Fix</th></tr>"
                
                for vuln in vulns:
                    vuln_tables_html += f"""
                    <tr>
                        <td>{vuln.get('type', '')}</td>
                        <td>{vuln.get('package', '')}</td>
                        <td>{vuln.get('vulnerability', '')}</td>
                        <td>{vuln.get('description', '')[:100]}...</td>
                        <td>{vuln.get('fixed_in', 'N/A')}</td>
                    </tr>
                    """
                
                vuln_tables_html += "</table>"
        
        # Format HTML
        html_content = html_template.format(
            scan_date=self.scan_results["scan_date"],
            total=self.scan_results["summary"]["total_vulnerabilities"],
            critical=self.scan_results["summary"]["critical"],
            high=self.scan_results["summary"]["high"],
            medium=self.scan_results["summary"]["medium"],
            low=self.scan_results["summary"]["low"],
            recommendations=recommendations_html,
            vulnerability_tables=vuln_tables_html
        )
        
        # Save HTML report
        html_path = self.project_root / "security-scan-report.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"📄 HTML report saved to: {html_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="KnowledgeHub Security Dependency Scanner"
    )
    parser.add_argument(
        "--path",
        default=".",
        help="Project path to scan (default: current directory)"
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "html", "both"],
        default="both",
        help="Output format for report"
    )
    parser.add_argument(
        "--severity-threshold",
        choices=["critical", "high", "medium", "low"],
        default="low",
        help="Minimum severity to report"
    )
    
    args = parser.parse_args()
    
    # Run scanner
    scanner = SecurityScanner(Path(args.path))
    results = scanner.run_full_scan()
    
    # Exit with error code if critical vulnerabilities found
    if results["summary"]["critical"] > 0:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()