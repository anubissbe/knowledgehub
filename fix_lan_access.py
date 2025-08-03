#!/usr/bin/env python3
"""Fix LAN access issues for KnowledgeHub"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.security.monitoring import security_monitor

# Unblock the LAN IP that was incorrectly blocked
lan_ip = "192.168.1.158"
if security_monitor.is_ip_blocked(lan_ip):
    security_monitor.unblock_ip(lan_ip)
    print(f"✅ Unblocked IP: {lan_ip}")
else:
    print(f"ℹ️  IP {lan_ip} was not blocked")

# Check and report on blocked IPs
print(f"\n📊 Currently blocked IPs: {security_monitor.blocked_ips}")
print(f"📊 Currently suspicious IPs: {security_monitor.suspicious_ips}")

# Unblock all LAN IPs (192.168.1.x range)
lan_ips_to_unblock = []
for ip in security_monitor.blocked_ips:
    if ip.startswith("192.168.1."):
        lan_ips_to_unblock.append(ip)

for ip in lan_ips_to_unblock:
    security_monitor.unblock_ip(ip)
    print(f"✅ Unblocked LAN IP: {ip}")

print("\n✅ LAN access fix complete!")
print("\n💡 To prevent future blocking of LAN IPs, the security monitoring middleware should be updated to whitelist the 192.168.1.x range.")