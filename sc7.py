#!/usr/bin/env python3
"""
Automated testing script for NUVYOLO
Tests different confidence thresholds and YOLO models
"""

import subprocess
import requests
import time
import json
import os
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional
import logging
import threading
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("test_automation")

class NUVYOLOTester:
    def __init__(self):
        self.nuvyolo_process = None
        self.event_counter_process = None
        self.ffmpeg_process = None
        self.results = []
        self.test_duration = 180  # 3 minutes per test
        self.added_hostname_mapping = False  # Track if we added hostname mapping
        
        # Test configurations
        self.confidence_levels = [0.7]
        self.models = ["yolov8n.pt"]
        self.trackers = ["botsort.yaml"]
        self.camera_id = 51  # Default camera ID for testing
        
        # FFmpeg stream configuration
        self.video_file_path = "/home/thiago/Git/nuvbash/prepared.flv"
        self.rtmp_url = f"rtmp://localhost/stream/{self.camera_id}"
        
        # API endpoints
        self.nuvyolo_base_url = "http://localhost:8000"
        self.event_counter_url = "http://localhost:8080"
        
        # Check if we need to use 'event-viewer' hostname instead of localhost
        self.event_counter_host = "localhost"  # Default

    def check_event_url_configuration(self):
        """Check what event URL NUVYOLO is configured to use"""
        env_file_path = ".env"
        if os.path.exists(env_file_path):
            try:
                with open(env_file_path, 'r') as f:
                    env_content = f.read()
                    if 'event-viewer' in env_content:
                        logger.warning("⚠️  Detected 'event-viewer' hostname in .env file")
                        logger.info("💡 Consider updating SEND_EVENT_URL to use localhost:8080")
                        self.event_counter_host = "event-viewer"
                        self.event_counter_url = "http://event-viewer:8080"
                        return True
            except Exception as e:
                logger.warning(f"Could not read .env file: {e}")
        return False
        
    def cleanup_processes(self):
        """Clean up any running processes"""
        logger.info("Cleaning up processes...")
        
        # Clean up hostname mapping if we added it
        if hasattr(self, 'added_hostname_mapping') and self.added_hostname_mapping:
            try:
                logger.info("🧹 Removing temporary hostname mapping...")
                remove_command = "sudo sed -i '/127.0.0.1 event-viewer/d' /etc/hosts"
                subprocess.run(remove_command, shell=True)
                logger.info("✅ Hostname mapping removed")
            except Exception as e:
                logger.warning(f"⚠️  Could not remove hostname mapping: {e}")
        
        # Stop NUVYOLO first
        if self.nuvyolo_process:
            try:
                logger.info("🛑 Stopping NUVYOLO...")
                self.nuvyolo_process.terminate()
                self.nuvyolo_process.wait(timeout=10)
                logger.info("✅ NUVYOLO stopped")
            except subprocess.TimeoutExpired:
                logger.warning("⏰ NUVYOLO didn't respond, force killing...")
                self.nuvyolo_process.kill()
                self.nuvyolo_process.wait()
                logger.info("✅ NUVYOLO force killed")
            except Exception as e:
                logger.error(f"❌ Error terminating NUVYOLO process: {e}")
        
        # Stop FFmpeg stream
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=5)
                logger.info("FFmpeg stream stopped")
            except subprocess.TimeoutExpired:
                self.ffmpeg_process.kill()
                self.ffmpeg_process.wait()
                logger.info("FFmpeg stream forcefully killed")
            except Exception as e:
                logger.error(f"Error terminating FFmpeg process: {e}")
        
        # Stop event counter last
        if self.event_counter_process:
            try:
                self.event_counter_process.terminate()
                self.event_counter_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.event_counter_process.kill()
                self.event_counter_process.wait()
            except Exception as e:
                logger.error(f"Error terminating event counter process: {e}")

    def start_event_counter(self):
        """Start the event counter server"""
        logger.info("Starting event counter server...")
        
        # Check if we need to handle hostname mapping
        uses_event_viewer_hostname = self.check_event_url_configuration()
        
        if uses_event_viewer_hostname:
            logger.info("🔧 Detected 'event-viewer' hostname configuration")
            logger.info("💡 Adding hostname mapping for event-viewer -> localhost")
            
            # Add hostname mapping temporarily
            try:
                # Check if mapping already exists
                with open('/etc/hosts', 'r') as f:
                    hosts_content = f.read()
                    
                if 'event-viewer' not in hosts_content:
                    # Add the mapping
                    mapping_command = 'echo "127.0.0.1 event-viewer" | sudo tee -a /etc/hosts'
                    result = subprocess.run(mapping_command, shell=True, capture_output=True, text=True)
                    if result.returncode == 0:
                        logger.info("✅ Added event-viewer hostname mapping")
                        self.added_hostname_mapping = True
                    else:
                        logger.warning("⚠️  Could not add hostname mapping automatically")
                        logger.info("💡 Please run: sudo echo '127.0.0.1 event-viewer' >> /etc/hosts")
                else:
                    logger.info("✅ event-viewer hostname mapping already exists")
                    
            except Exception as e:
                logger.warning(f"⚠️  Could not modify /etc/hosts: {e}")
                logger.info("💡 You may need to manually add '127.0.0.1 event-viewer' to /etc/hosts")
        
        # Save the event counter script
        event_counter_script = '''#!/usr/bin/env python3
import sys
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Tuple
import uvicorn
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("event_counter")

app = FastAPI(title="Event Counter")

class EventReceive(BaseModel):
    camera_id: int
    start: str
    end: str
    event_type: str
    tag: str
    coord_initial: Tuple[int, int]
    coord_end: Tuple[int, int]
    print: str

event_counters = {
    "total_events": 0,
    "events_by_tag": {},
    "events_by_camera": {},
    "test_session": None,
    "all_events": []
}

@app.post("/events/receive")
async def receive_event(event: EventReceive):
    try:
        event_counters["total_events"] += 1
        
        if event.tag not in event_counters["events_by_tag"]:
            event_counters["events_by_tag"][event.tag] = 0
        event_counters["events_by_tag"][event.tag] += 1
        
        if event.camera_id not in event_counters["events_by_camera"]:
            event_counters["events_by_camera"][event.camera_id] = 0
        event_counters["events_by_camera"][event.camera_id] += 1
        
        # Save event image
        import os
        os.makedirs("dashboard/images", exist_ok=True)
        
        event_id = f"event_{event_counters['total_events']}"
        image_filename = f"{event_id}.jpg"
        image_path = f"dashboard/images/{image_filename}"
        
        # Convert hex to bytes and save image
        try:
            image_bytes = bytes.fromhex(event.print)
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
        except Exception as img_error:
            logger.error(f"Error saving image: {img_error}")
            image_filename = None
        
        # Store detailed event data
        event_data = {
            "id": event_id,
            "camera_id": event.camera_id,
            "start": event.start,
            "end": event.end,
            "event_type": event.event_type,
            "tag": event.tag,
            "coord_initial": event.coord_initial,
            "coord_end": event.coord_end,
            "received_at": datetime.now().isoformat(),
            "test_session": event_counters.get("test_session"),
            "image_filename": image_filename
        }
        event_counters["all_events"].append(event_data)
        
        logger.info(f"Event #{event_counters['total_events']}: {event.tag} from camera {event.camera_id}")
        
        return {"status": "success", "total_count": event_counters["total_events"]}
    except Exception as e:
        logger.error(f"Error receiving event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    return event_counters

@app.get("/export_results")
async def export_results():
    return {
        "session_info": {
            "test_session": event_counters["test_session"],
            "total_events": event_counters["total_events"]
        },
        "summary": {
            "events_by_tag": event_counters["events_by_tag"],
            "events_by_camera": event_counters["events_by_camera"]
        },
        "all_events": event_counters["all_events"]
    }

@app.post("/reset")
async def reset_counters():
    global event_counters
    event_counters = {
        "total_events": 0,
        "events_by_tag": {},
        "events_by_camera": {},
        "test_session": None,
        "all_events": []
    }
    logger.info("Event counters reset")
    return {"status": "success"}

@app.post("/start_test_session")
async def start_test_session(session_name: str):
    logger.info(f"Starting test session: {session_name}")
    await reset_counters()
    event_counters["test_session"] = session_name
    logger.info(f"Test session started: {session_name}")
    return {"status": "success", "session": session_name}

@app.get("/")
async def root():
    return {
        "service": "Event Counter", 
        "total_events": event_counters["total_events"],
        "status": "running"
    }

if __name__ == "__main__":
    logger.info("Starting Event Counter server on port 8080...")
    try:
        uvicorn.run(
            "event_counter_simple:app", 
            host="0.0.0.0", 
            port=8080, 
            reload=False,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)
'''
        
        with open("event_counter_simple.py", "w") as f:
            f.write(event_counter_script)
        
        try:
            logger.info("🚀 Starting event counter process...")
            self.event_counter_process = subprocess.Popen([
                sys.executable, "event_counter_simple.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for server to start
            logger.info("⏳ Waiting for event counter to start...")
            time.sleep(5)  # Increased from 3 to 5 seconds
            
            # Test if server is running with more detailed checks
            max_retries = 10
            for i in range(max_retries):
                try:
                    response = requests.get(f"{self.event_counter_url}/", timeout=2)
                    if response.status_code == 200:
                        # Try to parse as JSON
                        try:
                            json_response = response.json()
                            logger.info("✅ Event counter server started successfully")
                            logger.info(f"📡 Server response: {json_response}")
                            return True
                        except ValueError as json_error:
                            logger.warning(f"⚠️  Server responding but not JSON: {response.text[:100]}")
                            if i == max_retries - 1:
                                # Last attempt, still not JSON - might be uvicorn startup message
                                logger.info("🔄 Server seems to be starting up, trying direct endpoint test...")
                                # Test a specific endpoint
                                try:
                                    stats_response = requests.get(f"{self.event_counter_url}/stats", timeout=2)
                                    if stats_response.status_code == 200:
                                        stats_json = stats_response.json()
                                        logger.info("✅ Event counter server started successfully (verified via /stats)")
                                        logger.info(f"📡 Stats response: {stats_json}")
                                        return True
                                except Exception:
                                    pass
                    else:
                        logger.warning(f"⚠️  Event counter responded with status {response.status_code}")
                except requests.exceptions.ConnectionError:
                    logger.info(f"⏳ Attempt {i+1}/{max_retries}: Event counter not ready yet...")
                    time.sleep(1)
                except Exception as e:
                    if "Expecting value" in str(e):
                        logger.info(f"⏳ Attempt {i+1}/{max_retries}: Server starting up (not JSON yet)...")
                    else:
                        logger.warning(f"⚠️  Error checking event counter: {e}")
                    time.sleep(1)
            
            # Check if process is still running
            if self.event_counter_process.poll() is not None:
                stdout, stderr = self.event_counter_process.communicate()
                logger.error("❌ Event counter process died:")
                logger.error(f"STDOUT: {stdout.decode()}")
                logger.error(f"STDERR: {stderr.decode()}")
                return False
            
            logger.error("❌ Event counter server failed to start properly within timeout")
            return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start event counter server: {e}")
            return False

    def reset_event_counter(self) -> bool:
        """Reset the event counter"""
        try:
            response = requests.post(f"{self.event_counter_url}/reset")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error resetting counter: {e}")
            return False

    def start_nuvyolo(self):
        """Start the NUVYOLO application"""
        logger.info("Starting NUVYOLO application...")
        
        try:
            self.nuvyolo_process = subprocess.Popen([
                "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for server to start
            time.sleep(5)
            
            # Test if server is running
            response = requests.get(f"{self.nuvyolo_base_url}/")
            if response.status_code == 200:
                logger.info("NUVYOLO application started successfully")
                return True
            else:
                logger.error("NUVYOLO application failed to start properly")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start NUVYOLO application: {e}")
            return False
    
    def stop_nuvyolo(self) -> bool:
        """Stop the NUVYOLO application"""
        logger.info("🛑 Stopping NUVYOLO application...")
        
        if self.nuvyolo_process:
            try:
                self.nuvyolo_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.nuvyolo_process.wait(timeout=10)
                    logger.info("✅ NUVYOLO stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't respond
                    logger.warning("⏰ NUVYOLO didn't respond to SIGTERM, forcing kill...")
                    self.nuvyolo_process.kill()
                    self.nuvyolo_process.wait()
                    logger.info("✅ NUVYOLO forcefully stopped")
                
                self.nuvyolo_process = None
                return True
                
            except Exception as e:
                logger.error(f"❌ Error stopping NUVYOLO: {e}")
                return False
        else:
            logger.info("ℹ️  No NUVYOLO process to stop")
            return True
    
    def restart_nuvyolo(self) -> bool:
        """Restart the NUVYOLO application"""
        logger.info("🔄 Restarting NUVYOLO application...")
        
        # Stop current instance
        if not self.stop_nuvyolo():
            logger.error("❌ Failed to stop NUVYOLO")
            return False
        
        # Wait a moment for cleanup
        time.sleep(2)
        
        # Start new instance
        if not self.start_nuvyolo():
            logger.error("❌ Failed to restart NUVYOLO")
            return False
        
        # Wait for service to be ready
        if not self.wait_for_nuvyolo_ready():
            logger.error("❌ NUVYOLO failed to become ready after restart")
            return False
        
        logger.info("✅ NUVYOLO restarted successfully")
        return True
    
    def wait_for_nuvyolo_ready(self, max_retries: int = 15) -> bool:
        """Wait for NUVYOLO to be ready"""
        logger.info("⏳ Waiting for NUVYOLO to be ready...")
        
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.nuvyolo_base_url}/", timeout=2)
                if response.status_code == 200:
                    logger.info("✅ NUVYOLO is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
            
        logger.error("❌ NUVYOLO failed to become ready within timeout")
        return False
    
    def wait_for_services(self):
        """Wait for both services to be ready"""
        logger.info("Waiting for services to be ready...")
        
        max_retries = 30
        for i in range(max_retries):
            try:
                nuvyolo_response = requests.get(f"{self.nuvyolo_base_url}/", timeout=2)
                counter_response = requests.get(f"{self.event_counter_url}/", timeout=2)
                
                if nuvyolo_response.status_code == 200 and counter_response.status_code == 200:
                    logger.info("Both services are ready")
                    return True
                    
            except requests.exceptions.RequestException:
                pass
                
            time.sleep(1)
            
        logger.error("Services failed to become ready within timeout")
        return False

    def start_video_stream(self) -> bool:
        """Start FFmpeg video stream"""
        logger.info(f"Starting video stream: {self.video_file_path} -> {self.rtmp_url}")
        
        # Check if video file exists
        if not os.path.exists(self.video_file_path):
            logger.error(f"Video file not found: {self.video_file_path}")
            return False
        
        ffmpeg_command = [
            "ffmpeg",
            "-re",                    # Read input at native frame rate
            "-stream_loop", "-1",     # Loop the input indefinitely
            "-i", self.video_file_path,  # Input file
            "-c:v", "copy",           # Copy video codec
            "-c:a", "copy",           # Copy audio codec
            "-f", "flv",              # Output format
            self.rtmp_url             # Output URL
        ]
        
        logger.info(f"🔧 FFmpeg command: {' '.join(ffmpeg_command)}")
        
        try:
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None  # For proper cleanup
            )
            
            # Give FFmpeg a moment to start and check output
            time.sleep(3)
            
            # Check if process is still running (not crashed immediately)
            if self.ffmpeg_process.poll() is None:
                logger.info("✅ Video stream started successfully")
                
                # Try to get some initial output to verify it's working
                try:
                    # Non-blocking read of stderr for initial status
                    import select
                    if select.select([self.ffmpeg_process.stderr], [], [], 0.1)[0]:
                        initial_output = self.ffmpeg_process.stderr.read(1024).decode('utf-8', errors='ignore')
                        if initial_output:
                            logger.info(f"📺 FFmpeg initial output: {initial_output[:200]}...")
                except:
                    pass  # Non-critical if we can't read output
                
                return True
            else:
                # Process died immediately, get error
                stdout, stderr = self.ffmpeg_process.communicate()
                logger.error(f"❌ FFmpeg failed to start:")
                logger.error(f"STDOUT: {stdout.decode()}")
                logger.error(f"STDERR: {stderr.decode()}")
                return False
                
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH")
            return False
        except Exception as e:
            logger.error(f"Failed to start video stream: {e}")
            return False
    
    def stop_video_stream(self) -> bool:
        """Stop FFmpeg video stream"""
        if not self.ffmpeg_process:
            logger.warning("No FFmpeg process to stop")
            return True
            
        logger.info("Stopping video stream...")
        
        try:
            # Try graceful termination first
            self.ffmpeg_process.terminate()
            
            # Wait for process to end
            try:
                self.ffmpeg_process.wait(timeout=5)
                logger.info("✅ Video stream stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't respond
                logger.warning("FFmpeg didn't respond to SIGTERM, forcing kill...")
                self.ffmpeg_process.kill()
                self.ffmpeg_process.wait()
                logger.info("✅ Video stream forcefully stopped")
            
            self.ffmpeg_process = None
            return True
            
        except Exception as e:
            logger.error(f"Error stopping video stream: {e}")
            return False
    
    def start_monitoring(self, confidence: float, model: str, tracker: str) -> bool:
        """Start monitoring with specific configuration"""
        logger.info(f"Starting monitoring with confidence={confidence}, model={model}, tracker={tracker}")
        
        payload = {
            "camera_id": self.camera_id,
            "device": "cuda",  # Change to "cpu" if you don't have GPU
            "detection_model_path": f"models/{model}",
            "classes": ["person", "car", "truck", "bus", "motorcycle", "bicycle"],
            "tracker_model": tracker,
            "frames_per_second": 10,
            "frames_before_disappearance": 30,
            "confidence_threshold": confidence,
            "iou": 0.45
        }
        
        logger.info(f"🔧 Monitoring payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = requests.post(
                f"{self.nuvyolo_base_url}/monitor",
                json=payload,
                timeout=30
            )
            
            logger.info(f"📡 Monitor API response status: {response.status_code}")
            logger.info(f"📡 Monitor API response: {response.text}")
            
            if response.status_code == 200:
                logger.info("✅ Monitoring started successfully")
                
                # Verify the camera is actually being monitored
                time.sleep(2)
                monitored_response = requests.get(f"{self.nuvyolo_base_url}/monitored")
                logger.info(f"📹 Monitored cameras: {monitored_response.text}")
                
                return True
            else:
                logger.error(f"❌ Failed to start monitoring: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop monitoring"""
        logger.info("🛑 Stopping monitoring...")
        
        try:
            response = requests.post(
                f"{self.nuvyolo_base_url}/stop/{self.camera_id}",
                timeout=15  # Add timeout to prevent hanging
            )
            
            logger.info(f"📡 Stop monitoring response: {response.status_code} - {response.text}")
            
            if response.status_code == 200:
                logger.info("✅ Monitoring stopped successfully")
                return True
            else:
                logger.error(f"❌ Failed to stop monitoring: {response.status_code} - {response.text}")
                # Try to force stop all monitoring as fallback
                logger.info("🔄 Attempting to force stop all monitoring...")
                try:
                    fallback_response = requests.post(
                        f"{self.nuvyolo_base_url}/stop/all",
                        timeout=10
                    )
                    logger.info(f"📡 Force stop response: {fallback_response.status_code}")
                    return fallback_response.status_code == 200
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback stop also failed: {fallback_error}")
                    return False
                
        except requests.exceptions.Timeout:
            logger.error("⏰ Timeout while stopping monitoring - NUVYOLO may be unresponsive")
            logger.info("🔄 Attempting force stop all as timeout recovery...")
            try:
                # Try force stop all with shorter timeout
                fallback_response = requests.post(
                    f"{self.nuvyolo_base_url}/stop/all",
                    timeout=5
                )
                logger.info(f"📡 Force stop response: {fallback_response.status_code}")
                return True  # Continue even if this fails
            except Exception:
                logger.warning("⚠️  Force stop also timed out - continuing anyway")
                return True  # Don't let this block the whole test
        except Exception as e:
            logger.error(f"❌ Error stopping monitoring: {e}")
            return False
    
    def get_monitored_cameras(self) -> Dict:
        """Get list of currently monitored cameras"""
        try:
            response = requests.get(f"{self.nuvyolo_base_url}/monitored")
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"❌ Failed to get monitored cameras: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"❌ Error getting monitored cameras: {e}")
            return {}
    
    def get_event_stats(self) -> Dict:
        """Get current event statistics"""
        try:
            response = requests.get(f"{self.event_counter_url}/stats")
            logger.info(f"📊 Event stats response: {response.status_code}")
            if response.status_code == 200:
                stats = response.json()
                logger.info(f"📊 Current stats: {json.dumps(stats, indent=2)}")
                return stats
            else:
                logger.error(f"❌ Failed to get stats: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {}
    
    def get_detailed_events(self) -> List[Dict]:
        """Get detailed events data from the event counter"""
        try:
            response = requests.get(f"{self.event_counter_url}/export_results")
            if response.status_code == 200:
                data = response.json()
                return data.get("all_events", [])
            else:
                logger.error(f"❌ Failed to get detailed events: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"❌ Error getting detailed events: {e}")
            return []

    def start_test_session(self, session_name: str) -> bool:
        """Start a new test session"""
        try:
            logger.info(f"🔄 Starting test session: {session_name}")
            logger.info(f"📡 Event counter URL: {self.event_counter_url}")
            
            response = requests.post(
                f"{self.event_counter_url}/start_test_session",
                params={"session_name": session_name},
                timeout=10
            )
            
            logger.info(f"📡 Test session response: {response.status_code}")
            logger.info(f"📡 Response content: {response.text}")
            
            if response.status_code == 200:
                logger.info("✅ Test session started successfully")
                return True
            else:
                logger.error(f"❌ Failed to start test session: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"❌ Connection error to event counter: {e}")
            logger.error("💡 Check if event counter server is running on port 8080")
            return False
        except requests.exceptions.Timeout as e:
            logger.error(f"❌ Timeout connecting to event counter: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error starting test session: {e}")
            return False
    
    def run_single_test(self, confidence: float, model: str, tracker: str) -> Dict:
        """Run a single test with given parameters"""
        test_name = f"conf_{confidence}_model_{model.replace('.pt', '')}_tracker_{tracker.replace('.yaml', '')}"
        logger.info(f"🧪 Running test: {test_name}")
        
        # Start test session
        if not self.start_test_session(test_name):
            logger.error(f"❌ Failed to start test session for {test_name}")
            return None
        
        # Start video stream
        logger.info("🎥 Starting video stream...")
        if not self.start_video_stream():
            logger.error(f"❌ Failed to start video stream for {test_name}")
            return None
        
        # Give stream time to stabilize
        logger.info("⏳ Waiting for stream to stabilize...")
        time.sleep(5)
        
        # Start monitoring
        if not self.start_monitoring(confidence, model, tracker):
            logger.error(f"❌ Failed to start monitoring for {test_name}")
            self.stop_video_stream()
            return None
        
        # Additional verification - check if monitoring is actually active
        time.sleep(3)
        monitored_cameras = self.get_monitored_cameras()
        logger.info(f"📹 Currently monitored cameras: {monitored_cameras}")
        
        # Check initial event counter state
        initial_stats = self.get_event_stats()
        logger.info(f"📊 Initial event count: {initial_stats.get('total_events', 0)}")
        
        # Wait for test duration with progress updates
        logger.info(f"🔍 Monitoring for {self.test_duration} seconds...")
        
        # Show progress during monitoring and check event count periodically
        for elapsed in range(0, self.test_duration, 10):
            time.sleep(10)
            remaining = self.test_duration - elapsed - 10
            
            # Check current event count
            current_stats = self.get_event_stats()
            current_count = current_stats.get('total_events', 0)
            
            if remaining > 0:
                logger.info(f"⏱️  {remaining} seconds remaining... Events so far: {current_count}")
            else:
                logger.info(f"⏱️  Final events: {current_count}")
        
        # Stop monitoring with timeout protection
        logger.info("⏹️  Stopping monitoring...")
        stop_success = False
        try:
            def stop_monitoring_thread(result_queue):
                try:
                    result = self.stop_monitoring()
                    result_queue.put(result)
                except Exception as e:
                    logger.error(f"Exception in stop_monitoring_thread: {e}")
                    result_queue.put(False)
            
            result_queue = queue.Queue()
            stop_thread = threading.Thread(target=stop_monitoring_thread, args=(result_queue,))
            stop_thread.daemon = True
            stop_thread.start()
            
            # Wait for thread to complete with timeout
            stop_thread.join(timeout=15)  # Reduced since we'll restart NUVYOLO anyway
            
            if stop_thread.is_alive():
                logger.error("⏰ Stop monitoring operation timed out after 15 seconds")
                logger.warning("🔄 Will restart NUVYOLO to ensure clean state...")
                stop_success = False
            else:
                try:
                    stop_success = result_queue.get_nowait()
                except:
                    stop_success = False
                    
        except Exception as e:
            logger.error(f"❌ Error during stop monitoring: {e}")
            stop_success = False
        
        if stop_success:
            logger.info("✅ Monitoring stopped successfully")
        else:
            logger.warning("⚠️  Monitoring stop failed or timed out")
        
        # Stop video stream
        logger.info("🛑 Stopping video stream...")
        self.stop_video_stream()
        
        # Always restart NUVYOLO after each test for clean state
        logger.info("🔄 Restarting NUVYOLO for clean state...")
        restart_success = self.restart_nuvyolo()
        
        if not restart_success:
            logger.error("❌ Failed to restart NUVYOLO - test results may be unreliable")
            return None
        
        # Small delay to ensure all events are processed
        logger.info("⏳ Waiting for final event processing...")
        time.sleep(2)
        
        # Get final stats and detailed events
        stats = self.get_event_stats()
        final_count = stats.get("total_events", 0)
        detailed_events = self.get_detailed_events()
        
        # Additional debugging info
        logger.info(f"🔍 Final analysis for {test_name}:")
        logger.info(f"   📊 Total events: {final_count}")
        logger.info(f"   📱 Events by tag: {stats.get('events_by_tag', {})}")
        logger.info(f"   📹 Events by camera: {stats.get('events_by_camera', {})}")
        
        # Compile test result
        result = {
            "test_name": test_name,
            "confidence": confidence,
            "model": model,
            "tracker": tracker,
            "duration_seconds": self.test_duration,
            "total_events": final_count,
            "events_by_tag": stats.get("events_by_tag", {}),
            "events_by_camera": stats.get("events_by_camera", {}),
            "detailed_events": detailed_events,
            "timestamp": datetime.now().isoformat(),
            "monitoring_stop_success": stop_success,
            "nuvyolo_restart_success": restart_success
        }
        
        # Save individual test data
        self.save_individual_test_data(result)
        
        if final_count > 0:
            logger.info(f"✅ Test {test_name} completed - Total events: {final_count}")
        else:
            logger.warning(f"⚠️  Test {test_name} completed with 0 events - check stream/detection configuration")
            
        return result
    
    def run_all_tests(self):
        """Run all test combinations"""
        logger.info("Starting automated testing...")
        total_combinations = len(self.confidence_levels) * len(self.models) * len(self.trackers)
        logger.info(f"Testing {len(self.confidence_levels)} confidence levels × {len(self.models)} models × {len(self.trackers)} trackers = {total_combinations} tests")
        
        current_test = 0
        
        for model in self.models:
            for tracker in self.trackers:
                for confidence in self.confidence_levels:
                    current_test += 1
                    logger.info(f"\n{'='*80}")
                    logger.info(f"TEST {current_test}/{total_combinations}: Model={model}, Tracker={tracker}, Confidence={confidence}")
                    logger.info(f"{'='*80}")
                    
                    result = self.run_single_test(confidence, model, tracker)
                    if result:
                        self.results.append(result)
                        logger.info(f"✅ Test completed successfully")
                    else:
                        logger.error(f"❌ Test failed")
                    
                    # Clean state between tests
                    if current_test < total_combinations:
                        logger.info("⏳ Waiting 5 seconds before next test...")
                        time.sleep(5)
    
    def save_individual_test_data(self, test_result: Dict) -> None:
        """Save individual test data to organized files"""
        try:
            # Create organized directory structure
            results_dir = "test_results"
            os.makedirs(results_dir, exist_ok=True)
            
            test_name = test_result["test_name"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed test data
            test_file = os.path.join(results_dir, f"{test_name}_{timestamp}.json")
            with open(test_file, "w") as f:
                json.dump(test_result, f, indent=2)
            
            logger.info(f"💾 Test data saved: {test_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving test data: {e}")

    def save_results(self):
        """Save test results to file and create dashboard"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nuvyolo_test_results_{timestamp}.json"
        
        # Create summary
        summary = {
            "test_summary": {
                "total_tests": len(self.results),
                "test_duration_per_test": self.test_duration,
                "confidence_levels": self.confidence_levels,
                "models": self.models,
                "trackers": self.trackers,
                "camera_id": self.camera_id,
                "video_file": self.video_file_path,
                "test_completed_at": datetime.now().isoformat()
            },
            "detailed_results": self.results
        }
        
        with open(filename, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        
        # Create dashboard
        self.create_dashboard(summary, timestamp)
        
        return filename
    
    def create_dashboard(self, summary: Dict, timestamp: str):
        """Create HTML dashboard for visualizing results"""
        try:
            dashboard_dir = "dashboard"
            os.makedirs(dashboard_dir, exist_ok=True)
            os.makedirs(os.path.join(dashboard_dir, "images"), exist_ok=True)
            
            # Create main dashboard HTML
            self.create_dashboard_html(summary, timestamp, dashboard_dir)
            
            # Create individual test pages
            self.create_test_detail_pages(summary, dashboard_dir)
            
            logger.info(f"📊 Dashboard created in {dashboard_dir}/")
            logger.info(f"🌐 Open {dashboard_dir}/index.html in your browser")
            
        except Exception as e:
            logger.error(f"❌ Error creating dashboard: {e}")
    
    def calculate_test_statistics(self, results: List[Dict]) -> Dict:
        """Calculate statistics for dashboard charts"""
        stats = {
            "total_events": sum(r["total_events"] for r in results),
            "avg_events": sum(r["total_events"] for r in results) / len(results) if results else 0,
            "by_confidence": {},
            "by_model": {},
            "by_tracker": {},
            "by_object_type": {}
        }
        
        for result in results:
            # By confidence
            conf = str(result["confidence"])
            stats["by_confidence"][conf] = stats["by_confidence"].get(conf, 0) + result["total_events"]
            
            # By model
            model = result["model"].replace(".pt", "")
            stats["by_model"][model] = stats["by_model"].get(model, 0) + result["total_events"]
            
            # By tracker
            tracker = result["tracker"].replace(".yaml", "")
            stats["by_tracker"][tracker] = stats["by_tracker"].get(tracker, 0) + result["total_events"]
            
            # By object type
            for obj_type, count in result["events_by_tag"].items():
                stats["by_object_type"][obj_type] = stats["by_object_type"].get(obj_type, 0) + count
        
        return stats
    
    def generate_test_list_html(self, results: List[Dict]) -> str:
        """Generate HTML for the test results list"""
        html = ""
        for result in results:
            events_class = "events-high" if result["total_events"] > 10 else "events-medium" if result["total_events"] > 5 else "events-low"
            
            html += f'''
        <div class="test-item" data-test-name="{result['test_name']}">
            <div class="test-name">{result['test_name']}</div>
            <div class="test-details">
                <div class="test-metric">Model: {result['model'].replace('.pt', '')}</div>
                <div class="test-metric">Tracker: {result['tracker'].replace('.yaml', '')}</div>
                <div class="test-metric">Confidence: {result['confidence']}</div>
                <div class="test-metric {events_class}">Events: {result['total_events']}</div>
                <div class="test-metric">Duration: {result['duration_seconds']//60}m {result['duration_seconds']%60}s</div>
            </div>
        </div>'''
        return html
    
    def create_dashboard_html(self, summary: Dict, timestamp: str, dashboard_dir: str):
        """Create the main dashboard HTML file"""
        
        # Calculate statistics
        stats = self.calculate_test_statistics(summary["detailed_results"])
        
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NUVYOLO Test Results Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f7fa;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border-left: 4px solid #667eea;
        }}
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin: 0;
        }}
        .stat-label {{
            color: #666;
            margin: 5px 0 0 0;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }}
        .chart-container {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}
        .chart-title {{
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 20px;
            color: #333;
        }}
        .test-list {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            overflow: hidden;
        }}
        .test-list h2 {{
            background: #f8f9fa;
            margin: 0;
            padding: 25px;
            border-bottom: 1px solid #e9ecef;
            font-size: 1.5em;
            color: #333;
        }}
        .test-item {{
            padding: 20px 25px;
            border-bottom: 1px solid #e9ecef;
            transition: background-color 0.3s ease;
            cursor: pointer;
        }}
        .test-item:hover {{
            background-color: #f8f9fa;
        }}
        .test-item:last-child {{
            border-bottom: none;
        }}
        .test-name {{
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
        }}
        .test-details {{
            color: #666;
            font-size: 0.9em;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }}
        .test-metric {{
            background: #f1f3f4;
            padding: 5px 10px;
            border-radius: 5px;
            text-align: center;
        }}
        .events-high {{ background-color: #d4edda; color: #155724; }}
        .events-medium {{ background-color: #fff3cd; color: #856404; }}
        .events-low {{ background-color: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 NUVYOLO Test Results Dashboard</h1>
        <p>Comprehensive analysis of object detection performance across models, trackers, and confidence levels</p>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | <strong>Total Tests:</strong> {len(summary["detailed_results"])}</p>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number">{len(summary["detailed_results"])}</div>
            <div class="stat-label">Total Tests</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{stats["total_events"]}</div>
            <div class="stat-label">Total Events Detected</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{stats["avg_events"]:.1f}</div>
            <div class="stat-label">Average Events per Test</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{len(summary["test_summary"]["models"])}</div>
            <div class="stat-label">Models Tested</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{len(summary["test_summary"]["trackers"])}</div>
            <div class="stat-label">Trackers Tested</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{summary["test_summary"]["test_duration_per_test"]//60}m</div>
            <div class="stat-label">Test Duration</div>
        </div>
    </div>

    <div class="charts-grid">
        <div class="chart-container">
            <div class="chart-title">📊 Events by Confidence Level</div>
            <canvas id="confidenceChart"></canvas>
        </div>
        <div class="chart-container">
            <div class="chart-title">🤖 Events by Model</div>
            <canvas id="modelChart"></canvas>
        </div>
        <div class="chart-container">
            <div class="chart-title">🎯 Events by Tracker</div>
            <canvas id="trackerChart"></canvas>
        </div>
        <div class="chart-container">
            <div class="chart-title">🚗 Object Types Distribution</div>
            <canvas id="objectChart"></canvas>
        </div>
    </div>

    <div class="test-list">
        <h2>📋 Detailed Test Results</h2>
        {self.generate_test_list_html(summary["detailed_results"])}
    </div>

    <script>
        // Chart.js configurations
        const chartOptions = {{
            responsive: true,
            plugins: {{
                legend: {{ position: 'top' }},
                tooltip: {{ mode: 'index', intersect: false }}
            }},
            scales: {{
                y: {{ beginAtZero: true }}
            }}
        }};

        // Confidence Chart
        const confidenceData = {json.dumps(stats["by_confidence"])};
        new Chart(document.getElementById('confidenceChart'), {{
            type: 'bar',
            data: {{
                labels: Object.keys(confidenceData),
                datasets: [{{
                    label: 'Total Events',
                    data: Object.values(confidenceData),
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 2
                }}]
            }},
            options: chartOptions
        }});

        // Model Chart
        const modelData = {json.dumps(stats["by_model"])};
        new Chart(document.getElementById('modelChart'), {{
            type: 'bar',
            data: {{
                labels: Object.keys(modelData),
                datasets: [{{
                    label: 'Total Events',
                    data: Object.values(modelData),
                    backgroundColor: 'rgba(118, 75, 162, 0.8)',
                    borderColor: 'rgba(118, 75, 162, 1)',
                    borderWidth: 2
                }}]
            }},
            options: chartOptions
        }});

        // Tracker Chart
        const trackerData = {json.dumps(stats["by_tracker"])};
        new Chart(document.getElementById('trackerChart'), {{
            type: 'doughnut',
            data: {{
                labels: Object.keys(trackerData),
                datasets: [{{
                    data: Object.values(trackerData),
                    backgroundColor: ['rgba(255, 99, 132, 0.8)', 'rgba(54, 162, 235, 0.8)'],
                    borderColor: ['rgba(255, 99, 132, 1)', 'rgba(54, 162, 235, 1)'],
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ position: 'top' }}
                }}
            }}
        }});

        // Object Types Chart
        const objectData = {json.dumps(stats["by_object_type"])};
        new Chart(document.getElementById('objectChart'), {{
            type: 'pie',
            data: {{
                labels: Object.keys(objectData),
                datasets: [{{
                    data: Object.values(objectData),
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.8)',
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(255, 205, 86, 0.8)',
                        'rgba(75, 192, 192, 0.8)',
                        'rgba(153, 102, 255, 0.8)',
                        'rgba(255, 159, 64, 0.8)'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ position: 'right' }}
                }}
            }}
        }});

        // Test item click handlers
        document.querySelectorAll('.test-item').forEach(item => {{
            item.addEventListener('click', function() {{
                const testName = this.dataset.testName;
                window.open(`test_details_${{testName}}.html`, '_blank');
            }});
        }});
    </script>
</body>
</html>'''
        
        with open(os.path.join(dashboard_dir, "index.html"), "w") as f:
            f.write(html_content)

    def create_test_detail_pages(self, summary: Dict, dashboard_dir: str):
        """Create individual test detail pages"""
        for result in summary["detailed_results"]:
            self.create_individual_test_page(result, dashboard_dir)
    
    def create_individual_test_page(self, test_result: Dict, dashboard_dir: str):
        """Create detailed page for individual test"""
        test_name = test_result["test_name"]
        
        # Process events for timeline
        events_timeline = []
        for event in test_result.get("detailed_events", []):
            events_timeline.append({
                "id": event.get("id", ""),
                "timestamp": event.get("received_at", ""),
                "type": event.get("tag", "unknown"),
                "camera": event.get("camera_id", ""),
                "start": event.get("start", ""),
                "end": event.get("end", ""),
                "coord_initial": event.get("coord_initial", [0, 0]),
                "coord_end": event.get("coord_end", [0, 0]),
                "image_filename": event.get("image_filename", None)
            })
        
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Details: {test_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f7fa;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        .back-button {{
            display: inline-block;
            padding: 10px 20px;
            background: rgba(255,255,255,0.2);
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin-bottom: 20px;
            transition: background 0.3s;
        }}
        .back-button:hover {{
            background: rgba(255,255,255,0.3);
        }}
        .test-config {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin-bottom: 30px;
        }}
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .config-item {{
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .config-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .config-value {{
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }}
        .events-section {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin-bottom: 30px;
        }}
        .events-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            max-height: 600px;
            overflow-y: auto;
        }}
        .event-card {{
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .event-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        .event-image {{
            width: 100%;
            height: 200px;
            object-fit: cover;
            background: #f8f9fa;
        }}
        .event-details {{
            padding: 15px;
        }}
        .event-title {{
            font-size: 1.1em;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }}
        .event-type {{
            display: inline-block;
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            margin-bottom: 8px;
        }}
        .event-car {{ background: #d4edda; color: #155724; }}
        .event-truck {{ background: #fff3cd; color: #856404; }}
        .event-person {{ background: #cce5ff; color: #004085; }}
        .event-bus {{ background: #f8d7da; color: #721c24; }}
        .event-motorcycle {{ background: #e7d4ff; color: #6a1b9a; }}
        .event-bicycle {{ background: #fff0e6; color: #bf6900; }}
        .event-meta {{
            font-size: 0.9em;
            color: #666;
            line-height: 1.4;
        }}
        .event-coords {{
            background: #f1f3f4;
            padding: 5px 8px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.8em;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <a href="index.html" class="back-button">← Back to Dashboard</a>
        <h1>🔍 Test Details: {test_name}</h1>
        <p>Detailed analysis of individual test run</p>
    </div>

    <div class="test-config">
        <h2>⚙️ Test Configuration</h2>
        <div class="config-grid">
            <div class="config-item">
                <div class="config-label">Model</div>
                <div class="config-value">{test_result['model'].replace('.pt', '')}</div>
            </div>
            <div class="config-item">
                <div class="config-label">Tracker</div>
                <div class="config-value">{test_result['tracker'].replace('.yaml', '')}</div>
            </div>
            <div class="config-item">
                <div class="config-label">Confidence</div>
                <div class="config-value">{test_result['confidence']}</div>
            </div>
            <div class="config-item">
                <div class="config-label">Total Events</div>
                <div class="config-value">{test_result['total_events']}</div>
            </div>
            <div class="config-item">
                <div class="config-label">Duration</div>
                <div class="config-value">{test_result['duration_seconds']//60}m {test_result['duration_seconds']%60}s</div>
            </div>
            <div class="config-item">
                <div class="config-label">Timestamp</div>
                <div class="config-value">{test_result['timestamp'][:19].replace('T', ' ')}</div>
            </div>
        </div>
    </div>

    <div class="events-section">
        <h2>📸 Detected Events ({len(events_timeline)})</h2>
        <div class="events-grid">'''
        
        # Add event cards
        for event in events_timeline:
            event_type_class = f"event-{event['type'].lower()}"
            image_src = f"images/{event['image_filename']}" if event['image_filename'] else "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='200'%3E%3Crect width='100%25' height='100%25' fill='%23f8f9fa'/%3E%3Ctext x='50%25' y='50%25' text-anchor='middle' dy='.3em' fill='%23666'%3ENo Image%3C/text%3E%3C/svg%3E"
            
            html_content += f'''
            <div class="event-card">
                <img src="{image_src}" alt="Event {event['id']}" class="event-image" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'300\\' height=\\'200\\'%3E%3Crect width=\\'100%25\\' height=\\'100%25\\' fill=\\'%23f8f9fa\\'/%3E%3Ctext x=\\'50%25\\' y=\\'50%25\\' text-anchor=\\'middle\\' dy=\\'.3em\\' fill=\\'%23666\\'%3EImage Error%3C/text%3E%3C/svg%3E'">
                <div class="event-details">
                    <div class="event-title">{event['id']}</div>
                    <span class="event-type {event_type_class}">{event['type'].upper()}</span>
                    <div class="event-meta">
                        <strong>Camera:</strong> {event['camera']}<br>
                        <strong>Time:</strong> {event['timestamp'][:19].replace('T', ' ')}<br>
                        <strong>Duration:</strong> {event['start']} - {event['end']}
                    </div>
                    <div class="event-coords">
                        Start: ({event['coord_initial'][0]}, {event['coord_initial'][1]})<br>
                        End: ({event['coord_end'][0]}, {event['coord_end'][1]})
                    </div>
                </div>
            </div>'''
        
        html_content += '''
        </div>
    </div>
</body>
</html>'''
        
        # Save the individual test page
        test_file_path = os.path.join(dashboard_dir, f"test_details_{test_name}.html")
        with open(test_file_path, "w") as f:
            f.write(html_content)


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("\n🛑 Received interrupt signal. Cleaning up...")
    if 'tester' in globals():
        tester.cleanup_processes()
    sys.exit(0)


def main():
    """Main execution function"""
    global tester
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("🚀 Starting NUVYOLO automated testing...")
    
    try:
        # Initialize tester
        tester = NUVYOLOTester()
        
        # Start event counter server
        logger.info("📊 Starting event counter server...")
        if not tester.start_event_counter():
            logger.error("❌ Failed to start event counter server")
            return
        
        # Start NUVYOLO application
        logger.info("🤖 Starting NUVYOLO application...")
        if not tester.start_nuvyolo():
            logger.error("❌ Failed to start NUVYOLO application")
            tester.cleanup_processes()
            return
        
        # Wait for services to be ready
        logger.info("⏳ Waiting for services to be ready...")
        if not tester.wait_for_services():
            logger.error("❌ Services failed to become ready")
            tester.cleanup_processes()
            return
        
        logger.info("✅ All services are ready!")
        
        # Run all tests
        logger.info("🧪 Starting test execution...")
        tester.run_all_tests()
        
        # Save results and create dashboard
        logger.info("💾 Saving results and creating dashboard...")
        results_file = tester.save_results()
        
        logger.info("🎉 Testing completed successfully!")
        logger.info(f"📄 Results saved to: {results_file}")
        logger.info("🌐 Open dashboard/index.html in your browser to view results")
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Testing interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always cleanup
        if 'tester' in locals():
            logger.info("🧹 Cleaning up processes...")
            tester.cleanup_processes()
        logger.info("✅ Cleanup completed")


if __name__ == "__main__":
    main()
