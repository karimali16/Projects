from scapy.all import *
from scapy.layers.inet import IP, TCP
import logging

# Set up logging
logging.basicConfig(filename='ids_log.txt', level=logging.INFO)

# List of suspicious IP addresses
SUSPICIOUS_IPS = ['192.168.90.11', '10.0.0.1']  # Add known malicious IPs here

def packet_callback(packet):
    """Callback function to process each captured packet."""
    if IP in packet:
        ip_src = packet[IP].src
        ip_dst = packet[IP].dst
        protocol = packet[IP].proto
        
        # Check for suspicious IP
        if ip_src in SUSPICIOUS_IPS or ip_dst in SUSPICIOUS_IPS:
            alert_message = f'Suspicious activity detected: {ip_src} to {ip_dst} using protocol {protocol}'
            print(alert_message)
            logging.warning(alert_message)

        # Example: Alert on TCP SYN packets (potential port scans)
        if TCP in packet and packet[TCP].flags == "S":
            alert_message = f'TCP SYN packet detected from {ip_src} to {ip_dst}'
            print(alert_message)
            logging.warning(alert_message)

def main():
    print("Starting IDS... Press Ctrl+C to stop.")
    
    # Start sniffing packets
    sniff(prn=packet_callback, store=0)

if __name__ == "__main__":
    main()