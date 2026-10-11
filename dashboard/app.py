
import os
import sys
import json
import csv
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from scoring_engine import SpeechEvaluator

class DashboardHandler(SimpleHTTPRequestHandler):
    evaluator = SpeechEvaluator(base_dir=PROJECT_ROOT)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.serve_file(os.path.join(CURRENT_DIR, "index.html"), "text/html")
        elif path == "/api/speeches":
            self.handle_speeches()
        elif path == "/api/evaluate":
            self.handle_evaluate(params)
        elif path == "/api/benchmark":
            self.handle_benchmark()
        elif path.startswith("/audio/"):
            audio_rel = path[len("/audio/"):]
            audio_full = os.path.join(PROJECT_ROOT, "audio", audio_rel)
            self.serve_audio_file(audio_full)
        elif path.startswith("/transcripts/"):
            t_id = path[len("/transcripts/"):]
            t_file = os.path.join(PROJECT_ROOT, "transcripts", f"{t_id}.txt")
            self.serve_file(t_file, "text/plain")
        else:
            super().do_GET()

    def serve_file(self, filepath, content_type):
        if not os.path.exists(filepath):
            self.send_error(404, "File not found")
            return
        with open(filepath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def serve_audio_file(self, filepath):
        if not os.path.exists(filepath):
            self.send_error(404, f"Audio file not found: {filepath}")
            return
        content_type = "audio/wav" if filepath.endswith(".wav") else "audio/mp4"
        with open(filepath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def handle_speeches(self):
        speeches_csv = os.path.join(PROJECT_ROOT, "metadata", "speeches.csv")
        speeches = []
        if os.path.exists(speeches_csv):
            with open(speeches_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    speeches.append({
                        "speech_id": r.get("speech_id"),
                        "title": r.get("title"),
                        "category": r.get("category"),
                        "speaker_id": r.get("speaker_id")
                    })
        self.send_json({"speeches": speeches})

    def handle_evaluate(self, params):
        speech_id = params.get("speech_id", ["speech_001"])[0]
        version = params.get("version", ["good"])[0]
        try:
            res = self.evaluator.evaluate_with_iou(speech_id, version)
            
            # Add audio relative URL if file exists
            audio_path = self.evaluator._get_audio_path(speech_id, version)
            if audio_path:
                res["audio_url"] = f"/audio/{speech_id}/{os.path.basename(audio_path)}"
            else:
                res["audio_url"] = None

            # Add transcript text
            t_path = os.path.join(PROJECT_ROOT, "transcripts", f"{speech_id}.txt")
            if os.path.exists(t_path):
                with open(t_path, "r", encoding="utf-8") as f:
                    content = f.read()
                raw = content.split("Transcript:", 1)[1].strip() if "Transcript:" in content else content
                res["transcript_text"] = raw
            else:
                res["transcript_text"] = ""

            self.send_json(res)
        except Exception as e:
            self.send_json({"error": str(e)}, status=500)

    def handle_benchmark(self):
        bench_path = os.path.join(PROJECT_ROOT, "evaluation", "benchmark_summary.json")
        if os.path.exists(bench_path):
            with open(bench_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.send_json(data)
        else:
            self.send_json({"error": "Benchmark results not yet generated. Run evaluation/evaluate_all.py first."}, status=404)

    def send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

def run_server(port=8080):
    server_address = ("", port)
    httpd = HTTPServer(server_address, DashboardHandler)
    print(f"============================================================")
    print(f"Speech Contrastive Analytics Dashboard Server")
    print(f"Dashboard running at: http://localhost:{port}")
    print(f"Press Ctrl+C to stop.")
    print(f"============================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dashboard HTTP Server")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    run_server(args.port)
