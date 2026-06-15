import http.server
import json
import os
import threading
from typing import Dict, Optional

class ViewerServer:
    def __init__(self, port: int = 8080):
        self.port = port
        self.lock = threading.Lock()
        
        # thread-safe cached data
        self.gaussians_bytes: bytes = b""
        self.live_image_bytes: bytes = b""
        self.live_depth_bytes: bytes = b""
        self.rendered_image_bytes: bytes = b""
        self.rendered_depth_bytes: bytes = b""
        self.poses_data: Dict = {
            "current_pose": None,
            "keyframes": []
        }
        
        self.config_data: Dict = {
            "is_training": False,
            "has_labels": False,
            "has_bboxes": False
        }
        
        # callbacks for dynamic data rendering
        self.gaussians_callback = None
        self.labels_callback = None
        
        self.render_mode = "rgb"
        
        # root directory for static frontend files
        self.static_dir = os.path.dirname(__file__)
        
        # keep reference to class instance for HTTP handler access
        ViewerServer.instance = self
        self.server: Optional[http.server.ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def update_data(self, gaussians_bytes: bytes, poses_data: Dict, live_image_bytes: bytes = b"", live_depth_bytes: bytes = b"", rendered_image_bytes: bytes = b"", rendered_depth_bytes: bytes = b""):
        with self.lock:
            self.gaussians_bytes = gaussians_bytes
            self.poses_data = poses_data
            if live_image_bytes:
                self.live_image_bytes = live_image_bytes
            if live_depth_bytes:
                self.live_depth_bytes = live_depth_bytes
            if rendered_image_bytes:
                self.rendered_image_bytes = rendered_image_bytes
            if rendered_depth_bytes:
                self.rendered_depth_bytes = rendered_depth_bytes

    def start(self):
        class ViewerHTTPHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")
                self.end_headers()

            def do_POST(self):
                cors_headers = [
                    ("Access-Control-Allow-Origin", "*"),
                    ("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                ]
                base_path = self.path.split("?")[0]
                if base_path == "/reason":
                    try:
                        content_length = int(self.headers.get('Content-Length', 0))
                        post_data = self.rfile.read(content_length)
                        
                        import urllib.request
                        req = urllib.request.Request("http://127.0.0.1:8081/reason", data=post_data, headers={'Content-Type': 'application/json'})
                        with urllib.request.urlopen(req) as response:
                            resp_data = response.read()
                            
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(resp_data)))
                        for header, val in cors_headers:
                            self.send_header(header, val)
                        self.end_headers()
                        try:
                            self.wfile.write(resp_data)
                        except BrokenPipeError:
                            pass
                    except Exception as e:
                        print(f"[ViewerServer] Proxy error to LLM: {e}")
                        self.send_error(500, str(e))
                else:
                    self.send_error(404, "Not Found")

            def do_GET(self):
                server_inst = ViewerServer.instance
                
                cors_headers = [
                    ("Access-Control-Allow-Origin", "*"),
                    ("Access-Control-Allow-Methods", "GET, OPTIONS"),
                    ("Cache-Control", "no-cache, no-store, must-revalidate")
                ]
                
                base_path = self.path.split("?")[0]
                
                if base_path == "/gaussians":
                    query = self.path.split("?")[1] if "?" in self.path else ""
                    from urllib.parse import parse_qs
                    params = parse_qs(query)
                    mode = params.get("mode", ["rgb"])[0]
                    server_inst.render_mode = mode

                    if server_inst.gaussians_callback:
                        data = server_inst.gaussians_callback(self.path)
                    else:
                        with server_inst.lock:
                            data = server_inst.gaussians_bytes
                    
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(data)))
                    for header, val in cors_headers:
                        self.send_header(header, val)
                    self.end_headers()
                    try:
                        self.wfile.write(data)
                    except BrokenPipeError:
                        pass # Client disconnected early, ignore
                    return
                    
                elif base_path == "/poses":
                    with server_inst.lock:
                        data = json.dumps(server_inst.poses_data).encode("utf-8")
                    
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    for header, val in cors_headers:
                        self.send_header(header, val)
                    self.end_headers()
                    self.wfile.write(data)
                    return

                elif base_path == "/live_image":
                    with server_inst.lock:
                        data = server_inst.live_image_bytes
                    
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    for header, val in cors_headers:
                        self.send_header(header, val)
                    self.end_headers()
                    self.wfile.write(data)
                    return

                elif base_path == "/live_depth":
                    with server_inst.lock:
                        data = server_inst.live_depth_bytes
                    
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    for header, val in cors_headers:
                        self.send_header(header, val)
                    self.end_headers()
                    self.wfile.write(data)
                    return

                elif base_path == "/rendered_image":
                    with server_inst.lock:
                        data = server_inst.rendered_image_bytes
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    for header, val in cors_headers:
                        self.send_header(header, val)
                    self.end_headers()
                    self.wfile.write(data)
                    return
                    
                elif base_path == "/rendered_depth":
                    with server_inst.lock:
                        data = server_inst.rendered_depth_bytes
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    for header, val in cors_headers:
                        self.send_header(header, val)
                    self.end_headers()
                    self.wfile.write(data)
                    return

                elif base_path == "/set_mode":
                    query = self.path.split("?")[1] if "?" in self.path else ""
                    from urllib.parse import parse_qs
                    params = parse_qs(query)
                    mode = params.get("mode", ["rgb"])[0]
                    server_inst.render_mode = mode
                    
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    for header, val in cors_headers:
                        self.send_header(header, val)
                    self.end_headers()
                    self.wfile.write(b'{"status":"ok"}')
                    return

                elif base_path == "/labels":
                    if server_inst.labels_callback:
                        data = server_inst.labels_callback()
                    else:
                        data = b'{"labels":[]}'
                    
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    for header, val in cors_headers:
                        self.send_header(header, val)
                    self.end_headers()
                    self.wfile.write(data)
                    return
                
                elif base_path == "/config":
                    with server_inst.lock:
                        data = json.dumps(server_inst.config_data).encode("utf-8")
                    
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    for header, val in cors_headers:
                        self.send_header(header, val)
                    self.end_headers()
                    self.wfile.write(data)
                    return
                
                rel_path = self.path.lstrip("/").split("?")[0]
                if not rel_path:
                    rel_path = "index.html"
                
                filepath = os.path.abspath(os.path.join(server_inst.static_dir, rel_path))
                if not filepath.startswith(os.path.abspath(server_inst.static_dir)):
                    self.send_error(403, "Forbidden")
                    return
                
                if rel_path.endswith(".html"):
                    content_type = "text/html"
                elif rel_path.endswith(".css"):
                    content_type = "text/css"
                elif rel_path.endswith(".js") or rel_path.endswith(".mjs"):
                    content_type = "application/javascript"
                elif rel_path.endswith(".ico"):
                    content_type = "image/x-icon"
                else:
                    content_type = "application/octet-stream"

                if os.path.exists(filepath) and os.path.isfile(filepath):
                    try:
                        with open(filepath, "rb") as f:
                            content = f.read()
                        self.send_response(200)
                        self.send_header("Content-Type", content_type)
                        self.send_header("Content-Length", str(len(content)))
                        for header, val in cors_headers:
                            self.send_header(header, val)
                        self.end_headers()
                        self.wfile.write(content)
                    except Exception as e:
                        self.send_error(500, f"Internal Server Error: {e}")
                else:
                    self.send_error(404, f"File {rel_path} not found")

        self.server = http.server.ThreadingHTTPServer(("0.0.0.0", self.port), ViewerHTTPHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"[ViewerServer] Web visualizer is active at http://localhost:{self.port}")

    def stop(self):
        """Shut down the HTTP server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("[ViewerServer] Server stopped successfully.")
