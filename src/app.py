from pathlib import Path

from flask import Flask, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from search import SearchIndex

DATA_DIR = Path(__file__).parent.parent / "data"
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

engine = SearchIndex()

DATA_DIR.mkdir(exist_ok=True)
files = sorted(
    (p for p in DATA_DIR.iterdir() if p.suffix.lower() in ALLOWED_EXTENSIONS),
    key=lambda p: p.stat().st_mtime,
)
if files:
    try:
        engine.load(files[-1])
    except ValueError as e:
        print(e)

app = Flask(__name__, template_folder=".", static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB


@app.route("/")
def search_page():
    query = request.args.get("q", "")
    message = request.args.get("message")
    results = engine.search(query) if query.strip() else []

    return render_template(
        "index.html", query=query, results=results, message=message, current_file=engine.filename
    )


@app.route("/style.css")
def style():
    return send_from_directory(app.root_path, "style.css")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return redirect(url_for("search_page", message="No file selected."))

    filename = secure_filename(file.filename)
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        return redirect(url_for("search_page", message="Only .pdf, .txt and .md files are allowed."))

    temp_path = DATA_DIR / f"upload_temp{Path(filename).suffix.lower()}"
    file.save(temp_path)
    try:
        engine.load(temp_path, filename)
    except ValueError:
        temp_path.unlink()
        return redirect(url_for("search_page", message=f"Could not read {filename}."))

    for old in DATA_DIR.iterdir():
        if old != temp_path and old.suffix.lower() in ALLOWED_EXTENSIONS:
            old.unlink()
    temp_path.rename(DATA_DIR / filename)

    return redirect(url_for("search_page"))


if __name__ == "__main__":
    app.run()
