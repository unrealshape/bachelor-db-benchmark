#!/usr/bin/env python3
"""Laedt Amazon Product Reviews und erzeugt lokale BGE-Embeddings im Thesis-Schema.

Datenquelle
-----------
`McAuley-Lab/Amazon-Reviews-2023` auf HuggingFace. Begruendung:

  - Direkter McAuley-Lineage (deckt sich mit der Thesis-Zitierung McAuley & Leskovec 2013).
  - Vollstaendige Felder, die das Thesis-Schema 5.1 deckt: `rating`, `title`,
    `text`, `asin` -> `product_id`, `user_id`, `timestamp`.
  - Die JSONL-Dateien liegen pro Kategorie unter `raw/review_categories/*.jsonl`
    und lassen sich per `hf_hub_url`/`hf_hub_download` direkt ziehen. Das
    umgeht die aktuelle `datasets`-Sperre fuer Loading-Scripts (>= 4.x), die
    `amazon_reviews_multi` und die `load_dataset`-Pfade fuer
    `McAuley-Lab/Amazon-Reviews-2023` blockiert.
  - Genug Volumen: alleine `Home_and_Kitchen` ~31 GB JSONL, `Books` ~20 GB,
    `Electronics` ~23 GB. Die Stufen S/M/L/XL/XXL koennen aus Kombinationen
    derselben Kategorien gespeist werden, ohne die Verteilung zu
    wechseln (Reviews und Queries kommen aus demselben Kategorien-Pool).

Embedding
---------
`BAAI/bge-large-en-v1.5` (MTEB-State-of-the-Art, MIT-Lizenz), 1024 dim,
L2-normalisiert. Laeuft lokal via `sentence-transformers` — kein API-Key,
keine Pro-Token-Kosten. Erstdownload des Modells ist ~1.3 GB nach
`~/.cache/huggingface/`.

Passages (Korpus-Texte) bekommen keine Instruction-Prefix. Die BGE-Instruction
"Represent this sentence for searching relevant passages: " wird ausschliesslich
bei Queries (`gen_queries.py`) vorangestellt.

Device-Wahl in dieser Reihenfolge: CUDA wenn verfuegbar, sonst MPS auf Apple
Silicon, sonst CPU. Batch-Size ueber `BENCH_EMBED_BATCH` (Default 64 fuer CPU,
256 fuer GPU/MPS).

Stufen (Thesis 5.1.3)
---------------------
Gemessen am Embedding-Volumen, nicht an der Dokumentenzahl. 1024 x float32
= 4096 Byte pro Vektor (reine Embeddings):

  S   ca. 10 GB   ->  ~2.62 Mio. Reviews
  M   ca. 20 GB   ->  ~5.24 Mio. Reviews
  L   ca. 40 GB   -> ~10.49 Mio. Reviews
  XL  ca. 80 GB   -> ~20.97 Mio. Reviews
  XXL ca. 100 GB  -> ~26.21 Mio. Reviews

`--stage` triggert die Vorgabe. Stufen sind unabhaengig (jede Stufe schreibt
ihr eigenes Verzeichnis); die kumulative Variante des alten Synthese-Loaders ist
fuer reale Reviews nicht praktikabel, weil ein 100 GB-Korpus einmal
geschrieben und dann fuer XL/L/M durch Subsetting genutzt werden koennte —
das bleibt offen, in der ersten Iteration wird pro Stufe komplett neu
geladen + embedded (Resume vermeidet doppelte Rechenzeit).

Resume
------
Cache unter `BENCH_CACHE_DIR` oder `~/.cache/bachelor-db-benchmark/reviewdata/`.
Pro Stufe ein Unterverzeichnis mit:

  - chunk_<NNNN>.parquet         (geschriebene Korpus-Chunks)
  - .progress.json               (zuletzt verarbeitete Quelle + Offset)
  - corpus_meta.json             (final, nach Abschluss)

Resume-Verhalten:
  * Existierende Chunks werden uebernommen.
  * Aus `.progress.json` wird der naechste zu verarbeitende Review-Index
    ermittelt; der Stream springt mittels Skip vor.
  * Wenn die Ziel-Groesse schon erreicht ist, schreibt der Loader nur die
    `corpus_meta.json` neu und beendet.
  * Findet sich im Cache ein altes Korpus mit anderer Embedding-Dimension
    (z. B. aus der vorigen OpenAI-Variante mit 1536 dim), bricht der
    Loader ab — keine automatische Ueberschreibung.

Aufruf
------
    # Dry-Run zeigt Plan, kein Modell-Download
    python load.py --stage S --dry-run

    # Echter Lauf
    python load.py --stage S
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Konstanten

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024
EMBEDDING_BYTES_PER_VEC = EMBEDDING_DIM * 4  # float32

# BGE-Query-Instruction: nur bei Queries, NICHT bei Passages.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Stufen — Embedding-Volumen (GB) -> Anzahl Reviews
STAGE_GB = {
    "S":   10,
    "M":   20,
    "L":   40,
    "XL":  80,
    "XXL": 100,
}


def stage_n_reviews(stage: str) -> int:
    gb = STAGE_GB[stage]
    return int(gb * (2 ** 30) / EMBEDDING_BYTES_PER_VEC)


# Default-Kategorien — gross genug fuer alle Stufen, mit breiter Domain-Streuung.
# Reihenfolge ist deterministisch, der Stream geht Kategorie fuer Kategorie.
DEFAULT_CATEGORIES = (
    "Home_and_Kitchen",
    "Clothing_Shoes_and_Jewelry",
    "Electronics",
    "Books",
    "Tools_and_Home_Improvement",
    "Health_and_Household",
    "Beauty_and_Personal_Care",
    "Sports_and_Outdoors",
    "Cell_Phones_and_Accessories",
    "Automotive",
    "Movies_and_TV",
    "Pet_Supplies",
    "Patio_Lawn_and_Garden",
    "Office_Products",
    "Toys_and_Games",
)

HF_REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
REVIEW_PATH_TEMPLATE = "raw/review_categories/{cat}.jsonl"
META_PATH_TEMPLATE = "raw/meta_categories/meta_{cat}.jsonl"

MIN_REVIEW_CHARS = 20


# ---------------------------------------------------------------------------
# Cache und Pfade

def cache_root() -> Path:
    env = os.environ.get("BENCH_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "bachelor-db-benchmark" / "reviewdata"


def stage_dir(stage: str) -> Path:
    return cache_root() / stage


# ---------------------------------------------------------------------------
# Device + Batch-Size

def pick_device() -> str:
    """CUDA > MPS (Apple Silicon) > CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_batch_size(device: str) -> int:
    env = os.environ.get("BENCH_EMBED_BATCH")
    if env:
        try:
            return int(env)
        except ValueError:
            print(f"  WARNUNG: BENCH_EMBED_BATCH={env!r} ist keine Zahl, nutze Default.",
                  flush=True)
    return 64 if device == "cpu" else 256


# ---------------------------------------------------------------------------
# CLI

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stage", required=True, choices=list(STAGE_GB))
    p.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Hartes Limit. Ueberschreibt den Stufen-Zielwert. "
             "Fuer Quick-Tests vor dem echten Stufenlauf gedacht.",
    )
    p.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        help="HF-Review-Kategorien (Reihenfolge = Stream-Reihenfolge)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Ueberschreibt den Default-Cache-Pfad (BENCH_CACHE_DIR/<stage>)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Reviews pro Embedding-Forward-Pass. "
             "Default kommt aus BENCH_EMBED_BATCH bzw. 64 (CPU) / 256 (GPU/MPS).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        choices=("cuda", "mps", "cpu"),
        help="Erzwingt ein Device. Default: auto (cuda > mps > cpu).",
    )
    p.add_argument(
        "--chunk-records",
        type=int,
        default=50_000,
        help="Reviews pro Parquet-Chunk",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=MIN_REVIEW_CHARS,
        help="Kuerzere review_text werden uebersprungen",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Trunkierung vor Embedding (BGE-Large: 512 Token Limit, ~2000 chars "
             "puffert konservativ).",
    )
    p.add_argument(
        "--meta",
        action="store_true",
        help="Produkt-Titel mitziehen (lazy per Kategorie). Default an.",
        default=True,
    )
    p.add_argument(
        "--no-meta",
        dest="meta",
        action="store_false",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan + Volumenschaetzung anzeigen, nichts schreiben, kein Modell-Download",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# HF-Downloads

def _hf_download(filename: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type="dataset",
    ))


def _open_jsonl(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def load_meta_titles(category: str) -> dict[str, str]:
    """ASIN -> product_title aus der Meta-JSONL einer Kategorie.

    Wird lazy aufgerufen, sobald die erste Review der Kategorie auftaucht.
    Bei grossen Kategorien kann das ein paar hundert MB RAM kosten — fuer
    die Stufen, mit denen die Thesis arbeitet (max. ~26 Mio. Reviews,
    O(1 Mio.) Produkte) ist das vertretbar."""
    fn = META_PATH_TEMPLATE.format(cat=category)
    print(f"  Meta-Titel laden: {fn} ...", flush=True)
    p = _hf_download(fn)
    titles: dict[str, str] = {}
    with _open_jsonl(p) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = d.get("parent_asin") or d.get("asin")
            title = d.get("title")
            if asin and title:
                titles[asin] = title
    print(f"    {len(titles):,} Titel", flush=True)
    return titles


# ---------------------------------------------------------------------------
# Review-Stream

@dataclass
class StreamProgress:
    """State fuer Resume."""
    category_index: int  # in args.categories
    review_offset: int   # 0-basierter Offset innerhalb der Kategorie
    n_written: int       # bereits in Chunks geschriebene Reviews
    next_chunk_idx: int  # naechster zu schreibender chunk-File-Index


def progress_path(out_dir: Path) -> Path:
    return out_dir / ".progress.json"


def load_progress(out_dir: Path) -> StreamProgress | None:
    pp = progress_path(out_dir)
    if not pp.exists():
        return None
    d = json.loads(pp.read_text())
    return StreamProgress(**d)


def save_progress(out_dir: Path, p: StreamProgress) -> None:
    progress_path(out_dir).write_text(json.dumps(p.__dict__, indent=2))


def stream_category(
    category: str,
    skip: int,
    titles: dict[str, str] | None,
    min_chars: int,
    max_chars: int,
):
    """Yields normalisierte Review-Dicts einer Kategorie.

    Springt die ersten `skip` Zeilen (vor Filterung). Skipping passiert auf
    Zeilenebene — das ist nicht 100% deckungsgleich mit unserer gefilterten
    Zaehlung, taugt aber als Resume-Approximation, weil der reine Linecount
    deterministisch ist.
    """
    fn = REVIEW_PATH_TEMPLATE.format(cat=category)
    print(f"  Stream: {fn} (skip={skip:,})", flush=True)
    p = _hf_download(fn)
    line_no = 0
    with _open_jsonl(p) as f:
        for line in f:
            if line_no < skip:
                line_no += 1
                continue
            line_no += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = d.get("text") or d.get("title") or ""
            if len(text) < min_chars:
                continue
            if len(text) > max_chars:
                text = text[:max_chars]
            asin = d.get("parent_asin") or d.get("asin") or ""
            user_id = d.get("user_id") or ""
            ts = d.get("timestamp")  # ms epoch
            if ts:
                try:
                    ts_iso = time.strftime(
                        "%Y-%m-%d", time.gmtime(int(ts) / 1000.0)
                    )
                except (ValueError, OSError):
                    ts_iso = ""
            else:
                ts_iso = ""
            yield {
                "product_id": asin,
                "product_title": (titles.get(asin, "") if titles else ""),
                "user_id": user_id,
                "rating": int(round(float(d.get("rating") or 0))),
                "review_text": text,
                "timestamp": ts_iso,
                "_category": category,
                "_line": line_no,  # 1-basiert nach Verarbeitung
            }


# ---------------------------------------------------------------------------
# Embedding

class BGEEmbedder:
    """Wraps BAAI/bge-large-en-v1.5 via sentence-transformers.

    Passages (Korpus) bekommen KEINE Instruction. Die Query-Instruction wird
    in `gen_queries.py` direkt am Query-Text vorangestellt — der Embedder
    selbst macht hier keinen Unterschied.
    """

    def __init__(self, batch_size: int | None = None, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            sys.exit(
                "Paket `sentence-transformers` fehlt. "
                "pip install sentence-transformers>=2.7 torch>=2.0"
            )
        self.device = device or pick_device()
        self.batch_size = batch_size or default_batch_size(self.device)
        print(f"  Embedding-Device: {self.device}, batch_size={self.batch_size}",
              flush=True)
        print(f"  Lade Modell {EMBEDDING_MODEL} ...", flush=True)
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=self.device)
        # BGE-Large hat 512 Token Limit
        self.model.max_seq_length = 512
        # fp16 auf CUDA: ~2x Durchsatz, vernachlaessigbarer Float-Unterschied
        # (Output wird eh L2-normalisiert + als float32 gespeichert). Konsistent
        # ueber alle Stufen auf dieser Maschine.
        if self.device == "cuda" and os.environ.get("BENCH_EMBED_FP32") != "1":
            self.model = self.model.half()

    def encode(self, texts: list[str]) -> np.ndarray:
        embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embs.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Parquet-Output

def write_chunk(
    out_dir: Path,
    idx: int,
    docs: list[dict],
    embeddings: np.ndarray,
    id_offset: int,
) -> Path:
    n = len(docs)
    flat = pa.array(embeddings.reshape(-1), type=pa.float32())
    emb_arr = pa.FixedSizeListArray.from_arrays(flat, EMBEDDING_DIM)
    cols = {
        "id":            pa.array(
            np.arange(id_offset, id_offset + n, dtype=np.int64),
        ),
        "product_id":    pa.array([d["product_id"] for d in docs],
                                  type=pa.string()),
        "product_title": pa.array([d["product_title"] for d in docs],
                                  type=pa.string()),
        "user_id":       pa.array([d["user_id"] for d in docs],
                                  type=pa.string()),
        "rating":        pa.array([d["rating"] for d in docs], type=pa.int8()),
        "review_text":   pa.array([d["review_text"] for d in docs],
                                  type=pa.string()),
        "timestamp":     pa.array([d["timestamp"] for d in docs],
                                  type=pa.string()),
        "embedding":     emb_arr,
    }
    table = pa.Table.from_arrays(list(cols.values()), names=list(cols.keys()))
    path = out_dir / f"chunk_{idx:04d}.parquet"
    pq.write_table(table, path, compression="zstd")
    return path


def write_meta(out_dir: Path, stage: str, n_written: int, n_chunks: int,
               categories: list[str]) -> None:
    meta = {
        "stage": stage,
        "target_gb": STAGE_GB[stage],
        "target_reviews": stage_n_reviews(stage),
        "n_reviews": n_written,
        "n_chunks": n_chunks,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "normalize": "l2",
        "source": HF_REPO_ID,
        "categories": categories,
        "schema": [
            "id:int64", "product_id:string", "product_title:string",
            "user_id:string", "rating:int8", "review_text:string",
            "timestamp:string",
            f"embedding:fixed_size_list<float32>[{EMBEDDING_DIM}]",
        ],
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "corpus_meta.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Cache-Dimensions-Check (verhindert Mix von altem 1536-dim mit neuem 1024-dim)

def check_existing_cache(out_dir: Path) -> None:
    """Bricht ab, wenn unter `out_dir` ein Korpus mit abweichender Dim liegt.

    Trifft typischerweise zu, wenn vorher die OpenAI-Variante (1536 dim) lief.
    Kein automatisches Ueberschreiben — die alten Daten koennten teuer
    erkauft worden sein.
    """
    if not out_dir.exists():
        return
    meta_path = out_dir / "corpus_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            return
        old_dim = meta.get("embedding_dim")
        old_model = meta.get("embedding_model")
        if old_dim and old_dim != EMBEDDING_DIM:
            sys.exit(
                f"Inkompatibler Cache unter {out_dir}: corpus_meta.json zeigt "
                f"embedding_dim={old_dim} (model={old_model!r}). Erwartet wird "
                f"{EMBEDDING_DIM} ({EMBEDDING_MODEL}).\n"
                f"Aktion erforderlich: Verzeichnis manuell verschieben oder "
                f"loeschen, z. B.\n"
                f"  mv {out_dir} {out_dir}.dim{old_dim}.bak\n"
                f"Kein automatisches Ueberschreiben — die alten Embeddings "
                f"koennten kostenpflichtig erzeugt worden sein."
            )
    # Heuristik: Erster Chunk vorhanden, aber keine Meta? Dimension am Schema
    # ablesen. Nur warnen, wenn klar abweichend.
    chunks = sorted(out_dir.glob("chunk_*.parquet"))
    if chunks and not meta_path.exists():
        try:
            schema = pq.read_schema(chunks[0])
            emb_field = schema.field("embedding")
            t = emb_field.type
            if hasattr(t, "list_size") and t.list_size != EMBEDDING_DIM:
                sys.exit(
                    f"Inkompatibler Cache unter {out_dir}: chunk_0000.parquet "
                    f"hat embedding-Dim {t.list_size}, erwartet {EMBEDDING_DIM}.\n"
                    f"Vermutlich Reste der alten OpenAI-Variante (1536 dim). "
                    f"Bitte Verzeichnis manuell sichern oder loeschen — kein "
                    f"automatisches Ueberschreiben."
                )
        except (KeyError, OSError, pa.ArrowInvalid):
            pass


# ---------------------------------------------------------------------------
# Dry-Run

def run_dry(args, n_target: int) -> None:
    out = args.output_dir or stage_dir(args.stage)
    device = args.device or pick_device()
    batch = args.batch_size or default_batch_size(device)
    print(f"\n[DRY-RUN] Stufe {args.stage}: Ziel ~{STAGE_GB[args.stage]} GB Embeddings")
    print(f"  -> {n_target:,} Reviews (bei {EMBEDDING_DIM} dim float32)")
    print(f"  Quelle:      {HF_REPO_ID}")
    print(f"  Modell:      {EMBEDDING_MODEL}  (lokal, kein API-Key)")
    print(f"  Device:      {device}")
    print(f"  Kategorien:  {len(args.categories)} ({', '.join(args.categories[:3])} ...)")
    print(f"  Output:      {out}")
    print(f"  Cache-Root:  {cache_root()}")
    print()
    emb_gb = n_target * EMBEDDING_BYTES_PER_VEC / 2**30
    overhead = 1.4  # text + metadata + parquet-zstd Overhead
    disk_gb = emb_gb * overhead
    print(f"  Reine Embedding-Bytes:  {emb_gb:.1f} GB")
    print(f"  Geschaetzter Plattenplatz (mit Metadaten/Parquet/zstd): ~{disk_gb:.1f} GB")
    print()
    n_batches = (n_target + batch - 1) // batch
    print(f"  Embedding-Batches (a {batch}): {n_batches:,}")
    print(f"  Parquet-Chunks    (a {args.chunk_records:,}): "
          f"{(n_target + args.chunk_records - 1) // args.chunk_records}")
    print()
    print("Kein API-Key noetig. Modell-Erstdownload ~1.3 GB nach ~/.cache/huggingface/.")
    print(f"Echter Lauf:  python {Path(__file__).name} --stage {args.stage}")


# ---------------------------------------------------------------------------
# Hauptpipeline

def main():
    args = parse_args()
    n_target = stage_n_reviews(args.stage)
    if args.max_records is not None and args.max_records > 0:
        n_target = min(n_target, args.max_records)
        print(f"[info] --max-records aktiv: Ziel auf {n_target:,} Reviews begrenzt")

    if args.dry_run:
        run_dry(args, n_target)
        return

    out = args.output_dir or stage_dir(args.stage)
    check_existing_cache(out)
    out.mkdir(parents=True, exist_ok=True)

    # Resume
    prog = load_progress(out)
    if prog is None:
        prog = StreamProgress(
            category_index=0, review_offset=0, n_written=0, next_chunk_idx=0,
        )
        print(f"Neuer Lauf nach {out}")
    else:
        print(f"Resume: {prog.n_written:,} Reviews schon vorhanden, "
              f"Kategorie #{prog.category_index} ({args.categories[prog.category_index] if prog.category_index < len(args.categories) else '(done)'}), "
              f"Line-Offset {prog.review_offset:,}")

    if prog.n_written >= n_target:
        print(f"Ziel bereits erreicht ({prog.n_written:,}/{n_target:,}). "
              f"Schreibe corpus_meta.json und beende.")
        n_chunks = len(sorted(out.glob("chunk_*.parquet")))
        write_meta(out, args.stage, prog.n_written, n_chunks, args.categories)
        return

    embedder = BGEEmbedder(batch_size=args.batch_size, device=args.device)

    # Buffer fuer den aktuellen Parquet-Chunk
    buf: list[dict] = []
    t0 = time.time()

    def flush_chunk():
        nonlocal buf, prog
        if not buf:
            return
        texts = [d["review_text"] for d in buf]
        embs = embedder.encode(texts)
        write_chunk(out, prog.next_chunk_idx, buf, embs, prog.n_written)
        prog.n_written += len(buf)
        prog.next_chunk_idx += 1
        save_progress(out, prog)
        rate = prog.n_written / max(time.time() - t0, 1e-6)
        print(f"    chunk {prog.next_chunk_idx - 1:>4}  "
              f"total={prog.n_written:>12,}/{n_target:,}  "
              f"{rate:>7,.0f} doc/s", flush=True)
        buf = []

    titles: dict[str, str] | None = None
    current_cat: str | None = None

    while prog.n_written < n_target and prog.category_index < len(args.categories):
        cat = args.categories[prog.category_index]
        if cat != current_cat:
            current_cat = cat
            titles = load_meta_titles(cat) if args.meta else None

        for doc in stream_category(
            cat, prog.review_offset, titles,
            args.min_chars, args.max_chars,
        ):
            buf.append(doc)
            if len(buf) >= args.chunk_records:
                # Track line offset before flush so resume picks up correctly
                prog.review_offset = doc["_line"]
                # Drop helper fields before serializing to parquet
                for d in buf:
                    d.pop("_category", None)
                    d.pop("_line", None)
                flush_chunk()
                if prog.n_written >= n_target:
                    break
        else:
            # Kategorie zu Ende — naechste
            prog.category_index += 1
            prog.review_offset = 0
            current_cat = None
            titles = None
            continue
        break  # break-from-inner -> auch ausserhalb stoppen

    # Letzten unvollstaendigen Chunk schreiben
    if buf and prog.n_written < n_target:
        for d in buf:
            d.pop("_category", None)
            d.pop("_line", None)
        flush_chunk()

    n_chunks = len(sorted(out.glob("chunk_*.parquet")))
    write_meta(out, args.stage, prog.n_written, n_chunks, args.categories)

    bytes_on_disk = sum(p.stat().st_size for p in out.glob("chunk_*.parquet"))
    print(f"\nFertig: {prog.n_written:,} Reviews in {n_chunks} Chunks, "
          f"{bytes_on_disk / 2**30:.2f} GiB on disk")
    if prog.n_written < n_target:
        print(f"WARNUNG: Ziel ({n_target:,}) nicht erreicht — Kategorien aus?")


if __name__ == "__main__":
    main()
