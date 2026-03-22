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
import re
import sys

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


def load_model(model_key):
    """Lade ein vortrainiertes Wortvektor-Modell über gensim."""
    import gensim.downloader as api

    model_name = AVAILABLE_MODELS.get(model_key, model_key)
    print(f"\nLade Modell '{model_name}'...")
    print("(Beim ersten Mal wird das Modell heruntergeladen, das kann etwas dauern.)\n")
    wv = api.load(model_name)
    print(f"Modell geladen: {len(wv)} Wörter, {wv.vector_size} Dimensionen.\n")
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
