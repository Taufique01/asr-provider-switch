import os
from flask import Flask, request, jsonify, render_template
from utils.video_utils import get_whisper_srt
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

WHISPER_PROVIDER = os.getenv("WHISPER_PROVIDER", "openai")

@app.route("/")
def index():
    return render_template("test-build-srt-client.html")

@app.route("/build_srt_file", methods=["POST"])
def build_srt_file():
    file = request.files.get("audio")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    audio_bytes = file.read()
    result = get_whisper_srt(file.filename, audio_bytes)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
