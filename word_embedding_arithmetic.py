#!/usr/bin/env python3
"""
Wort-Embedding-Arithmetik: Interaktiver Rechner für Wortvektor-Operationen.

Basierend auf dem Word2Vec-Konzept von Mikolov et al. (2013):
Wortvektoren kodieren semantische Beziehungen als lineare Richtungen,
sodass Vektorarithmetik sinnvolle Ergebnisse liefert.

Beispiele:
  king - man + woman = queen
  paris - france + germany = berlin
  hitler - deutschland + italien ≈ mussolini

Verwendung:
  python word_embedding_arithmetic.py [--model MODEL_NAME] [--topn N]
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
import zipfile

import numpy as np

AVAILABLE_MODELS = {
    "glove-100": "glove-wiki-gigaword-100",
    "glove-200": "glove-wiki-gigaword-200",
    "glove-300": "glove-wiki-gigaword-300",
    "word2vec": "word2vec-google-news-300",
    "glove-twitter-100": "glove-twitter-100",
    "fasttext": "fasttext-wiki-news-subwords-300",
}

DEFAULT_MODEL = "glove-100"


def _get_gensim_data_dir():
    """Ermittle das gensim-data Verzeichnis."""
    base = os.environ.get("GENSIM_DATA_DIR", os.path.join(os.path.expanduser("~"), "gensim-data"))
    os.makedirs(base, exist_ok=True)
    return base


def _download_with_resume(url, dest_path, expected_size=None, max_retries=5):
    """Download einer Datei mit Resume-Support und Retry bei Unterbrechung."""
    for attempt in range(1, max_retries + 1):
        existing_size = 0
        if os.path.exists(dest_path):
            existing_size = os.path.getsize(dest_path)

        if expected_size and existing_size >= expected_size:
            print(f"  Datei bereits vollständig heruntergeladen ({existing_size} bytes).")
            return True

        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            print(f"  Setze Download fort ab {existing_size / 1024 / 1024:.1f} MB (Versuch {attempt}/{max_retries})...")
        else:
            print(f"  Starte Download (Versuch {attempt}/{max_retries})...")

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                # Prüfe ob Server Range unterstützt
                if existing_size > 0 and response.status != 206:
                    # Server unterstützt kein Resume, von vorne anfangen
                    print("  Server unterstützt kein Resume, starte von vorne...")
                    existing_size = 0
                    mode = "wb"
                else:
                    mode = "ab" if existing_size > 0 else "wb"

                total_size = None
                content_length = response.headers.get("Content-Length")
                if content_length:
                    total_size = int(content_length) + existing_size

                downloaded = existing_size
                chunk_size = 1024 * 1024  # 1 MB chunks
                last_print = time.time()

                with open(dest_path, mode) as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Fortschritt alle 2 Sekunden anzeigen
                        now = time.time()
                        if now - last_print >= 2 or not chunk:
                            if total_size:
                                pct = downloaded / total_size * 100
                                bar_done = int(pct / 2)
                                bar = "=" * bar_done + "-" * (50 - bar_done)
                                print(f"\r  [{bar}] {pct:.1f}% {downloaded / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MB", end="", flush=True)
                            else:
                                print(f"\r  {downloaded / 1024 / 1024:.1f} MB heruntergeladen...", end="", flush=True)
                            last_print = now

                print()  # Neue Zeile nach Fortschrittsanzeige
                print("  Download abgeschlossen.")
                return True

        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            print(f"\n  Download unterbrochen: {e}")
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  Warte {wait}s vor erneutem Versuch...")
                time.sleep(wait)
            else:
                print(f"  Alle {max_retries} Versuche fehlgeschlagen.")
                return False

    return False


def _robust_download_model(model_name):
    """Lade ein gensim-Modell mit robustem Download (Resume + Retry)."""
    import gensim.downloader as api

    data_dir = _get_gensim_data_dir()
    model_dir = os.path.join(data_dir, model_name)

    # Prüfe ob Modell bereits geladen ist
    if os.path.exists(os.path.join(model_dir, "__init__.py")):
        print(f"  Modell bereits vorhanden in {model_dir}")
        return api.load(model_name)

    # Hole Modell-Info von gensim
    info = api.info()
    if model_name not in info["models"]:
        print(f"Unbekanntes Modell: {model_name}")
        print(f"Verfügbar: {', '.join(info['models'].keys())}")
        sys.exit(1)

    model_info = info["models"][model_name]
    file_size = model_info.get("file_size", 0)
    file_name = model_info.get("file_name", f"{model_name}.gz")
    checksum = model_info.get("checksum", None)

    # Download-URL
    base_url = "https://raw.githubusercontent.com/RaRe-Technologies/gensim-data/master"
    url = f"{base_url}/{model_name}/{file_name}"

    os.makedirs(model_dir, exist_ok=True)
    dest_path = os.path.join(model_dir, file_name)

    print(f"  Modell: {model_name}")
    print(f"  Größe: {file_size / 1024 / 1024:.1f} MB")

    success = _download_with_resume(url, dest_path, expected_size=file_size)
    if not success:
        print("\nDownload fehlgeschlagen. Tipps:")
        print("  - Prüfe deine Internetverbindung")
        print("  - Versuche es später erneut (der Download wird fortgesetzt)")
        print(f"  - Nutze ein kleineres Modell: --model glove-100")
        sys.exit(1)

    # Checksum prüfen
    if checksum:
        print("  Prüfe Checksum...")
        sha = hashlib.sha256()
        with open(dest_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(chunk)
        if sha.hexdigest() != checksum:
            print(f"  WARNUNG: Checksum stimmt nicht überein!")
            print(f"  Erwartet:  {checksum}")
            print(f"  Erhalten:  {sha.hexdigest()}")
            print(f"  Lösche fehlerhaften Download...")
            os.remove(dest_path)
            sys.exit(1)

    # Entpacken falls nötig
    if file_name.endswith(".zip"):
        print("  Entpacke Archiv...")
        with zipfile.ZipFile(dest_path, "r") as zf:
            zf.extractall(model_dir)

    # Markiere als vollständig
    with open(os.path.join(model_dir, "__init__.py"), "w") as f:
        f.write("")

    # Lade über gensim
    return api.load(model_name)


def load_model(model_key):
    """Lade ein vortrainiertes Wortvektor-Modell."""
    model_name = AVAILABLE_MODELS.get(model_key, model_key)
    print(f"\nLade Modell '{model_name}'...")
    print("(Beim ersten Mal wird das Modell heruntergeladen, das kann etwas dauern.)\n")

    try:
        wv = _robust_download_model(model_name)
    except Exception as e:
        print(f"\nRobuster Download fehlgeschlagen ({e}), versuche Standard-gensim...")
        import gensim.downloader as api
        wv = api.load(model_name)

    print(f"\nModell geladen: {len(wv)} Wörter, {wv.vector_size} Dimensionen.\n")
    return wv


def parse_expression(expr):
    """
    Parse eine Ausdruck wie 'king - man + woman' in positive und negative Wörter.

    Gibt zurück: (positive_words, negative_words)
    """
    expr = expr.strip()
    if not expr:
        return [], []

    # Tokenisiere: erster Term ist implizit positiv
    tokens = re.findall(r'[+\-]|[^\s+\-]+', expr)

    positive = []
    negative = []
    sign = "+"

    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token == "+":
            sign = "+"
        elif token == "-":
            sign = "-"
        else:
            word = token.lower()
            if sign == "+":
                positive.append(word)
            else:
                negative.append(word)

    return positive, negative


def compute_and_display(wv, expression, topn=5):
    """Berechne Vektorarithmetik und zeige die nächsten Nachbarn."""
    positive, negative = parse_expression(expression)

    if not positive and not negative:
        print("Leerer Ausdruck. Bitte gib Wörter mit +/- ein.")
        return

    # Prüfe ob alle Wörter im Vokabular sind
    all_words = positive + negative
    missing = [w for w in all_words if w not in wv]
    if missing:
        print(f"Nicht im Vokabular: {', '.join(missing)}")
        for w in missing:
            # Vorschläge finden
            try:
                similar = wv.most_similar(positive=[w[:3]], topn=3)
                suggestions = [s[0] for s in similar]
                print(f"  Meintest du vielleicht: {', '.join(suggestions)}?")
            except (KeyError, Exception):
                pass
        return

    # Berechne manuell für die Anzeige
    result_vec = np.zeros(wv.vector_size, dtype=np.float32)
    parts = []
    for w in positive:
        result_vec += wv[w]
        parts.append(f"vec({w})")
    for w in negative:
        result_vec -= wv[w]
        parts.append(f"- vec({w})")

    formula = " + ".join(parts[:len(positive)])
    if negative:
        formula += " " + " ".join(parts[len(positive):])

    print(f"\n  Berechnung: {formula}")
    print(f"  {'─' * 50}")

    # Finde nächste Nachbarn
    try:
        results = wv.most_similar(positive=positive, negative=negative, topn=topn)
    except Exception as e:
        print(f"  Fehler: {e}")
        return

    print(f"  Top {topn} Ergebnisse:\n")
    for i, (word, score) in enumerate(results, 1):
        bar_len = int(score * 30) if score > 0 else 0
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    {i}. {word:<20} {bar} {score:.4f}")

    # Zeige auch Ähnlichkeit zwischen Eingabewörtern
    if len(all_words) >= 2:
        print(f"\n  Ähnlichkeiten zwischen Eingabewörtern:")
        for i, w1 in enumerate(all_words):
            for w2 in all_words[i + 1:]:
                sim = wv.similarity(w1, w2)
                print(f"    {w1} ↔ {w2}: {sim:.4f}")
    print()


def show_help():
    """Zeige Hilfe-Text."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              Wort-Embedding-Arithmetik                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Gib einen Ausdruck mit Wörtern und +/- ein:                ║
║                                                              ║
║  Beispiele:                                                  ║
║    king - man + woman           → queen                      ║
║    paris - france + germany     → berlin                     ║
║    walking - walk + swim        → swimming                   ║
║    bigger - big + small         → smaller                    ║
║                                                              ║
║  Befehle:                                                    ║
║    sim wort1 wort2     Ähnlichkeit zwischen zwei Wörtern     ║
║    nearest wort        Nächste Nachbarn eines Wortes          ║
║    vec wort            Zeige den Vektor eines Wortes          ║
║    topn N              Ändere Anzahl der Ergebnisse           ║
║    hilfe               Diese Hilfe anzeigen                   ║
║    quit / exit         Programm beenden                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def handle_command(wv, line, topn):
    """Verarbeite Spezialbefehle. Gibt neuen topn-Wert zurück."""
    parts = line.strip().split()
    cmd = parts[0].lower()

    if cmd == "sim" and len(parts) >= 3:
        w1, w2 = parts[1].lower(), parts[2].lower()
        missing = [w for w in [w1, w2] if w not in wv]
        if missing:
            print(f"  Nicht im Vokabular: {', '.join(missing)}")
        else:
            sim = wv.similarity(w1, w2)
            print(f"\n  Ähnlichkeit({w1}, {w2}) = {sim:.4f}\n")
        return topn

    elif cmd == "nearest" and len(parts) >= 2:
        word = parts[1].lower()
        if word not in wv:
            print(f"  '{word}' nicht im Vokabular.")
        else:
            results = wv.most_similar(positive=[word], topn=topn)
            print(f"\n  Nächste Nachbarn von '{word}':\n")
            for i, (w, score) in enumerate(results, 1):
                print(f"    {i}. {w:<20} {score:.4f}")
            print()
        return topn

    elif cmd == "vec" and len(parts) >= 2:
        word = parts[1].lower()
        if word not in wv:
            print(f"  '{word}' nicht im Vokabular.")
        else:
            vec = wv[word]
            print(f"\n  Vektor für '{word}' ({len(vec)} Dimensionen):")
            print(f"  Norm: {np.linalg.norm(vec):.4f}")
            print(f"  Min: {vec.min():.4f}, Max: {vec.max():.4f}, Mean: {vec.mean():.4f}")
            # Zeige erste 10 Dimensionen
            preview = ", ".join(f"{v:.3f}" for v in vec[:10])
            print(f"  Erste 10 Werte: [{preview}, ...]\n")
        return topn

    elif cmd == "topn" and len(parts) >= 2:
        try:
            topn = int(parts[1])
            print(f"  Anzahl Ergebnisse auf {topn} gesetzt.\n")
        except ValueError:
            print("  Ungültige Zahl.")
        return topn

    elif cmd in ("hilfe", "help", "?"):
        show_help()
        return topn

    return None  # Kein Befehl erkannt


def main():
    parser = argparse.ArgumentParser(
        description="Interaktiver Wort-Embedding-Arithmetik-Rechner"
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Modell-Name. Verfügbar: {', '.join(AVAILABLE_MODELS.keys())} "
             f"(Standard: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--topn", "-n",
        type=int,
        default=5,
        help="Anzahl der nächsten Nachbarn (Standard: 5)"
    )
    args = parser.parse_args()

    # Prüfe ob gensim installiert ist
    try:
        import gensim  # noqa: F401
    except ImportError:
        print("gensim ist nicht installiert. Installiere mit:")
        print("  pip install gensim")
        sys.exit(1)

    wv = load_model(args.model)
    topn = args.topn

    show_help()

    print("Bereit! Gib einen Ausdruck ein (oder 'hilfe' für Hilfe):\n")

    while True:
        try:
            line = input("🔤 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAuf Wiedersehen!")
            break

        if not line:
            continue

        if line.lower() in ("quit", "exit", "q"):
            print("Auf Wiedersehen!")
            break

        # Prüfe auf Spezialbefehle
        result = handle_command(wv, line, topn)
        if result is not None:
            topn = result
            continue

        # Normaler Arithmetik-Ausdruck
        compute_and_display(wv, line, topn)


if __name__ == "__main__":
    main()
