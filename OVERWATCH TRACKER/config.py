"""
OVERWATCH TRACKER Configuration
Edit these settings to match your setup
"""

# Camera Settings
CAMERA_IP = "192.168.1.111"  # Your phone's IP address
CAMERA_PORT = 8080            # IP Webcam default port

# You can also use a local webcam by setting this to a number (0, 1, 2...)
# CAMERA_SOURCE = 0  # Uncomment to use local webcam

# Trail Settings
TRAIL_LENGTH = 20  # Number of positions to keep for head trail

# Threat Level Thresholds
THREAT_ALERT = 25      # Yellow threshold
THREAT_DANGER = 50     # Orange threshold  
THREAT_CRITICAL = 70   # Red threshold
