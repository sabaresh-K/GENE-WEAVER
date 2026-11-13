# GeneWeaver: GPU-Accelerated CRISPR Alignment Engine

GeneWeaver is a high-throughput, memory-efficient CRISPR guide RNA (gRNA) alignment engine designed for large-scale genomic reference analysis (e.g., human genome GRCh38/T2T). It bridges chunked I/O genomic streaming with accelerated sequence matching algorithms (CPU sliding-window baseline, scaling to GPU/CUDA Numba kernels).

---

## 📁 Project Architecture

```
geneweaver-project/
├── geneweaver/
│   ├── __init__.py           # Package namespace
│   ├── parser.py             # BioPython streaming chunker (bytearray zero-copy buffers)
│   ├── utils.py              # Synthetic chromosome FASTA generator with CRISPR target injection
│   ├── baseline_cpu.py       # Pure Python sliding-window search & Hamming distance baseline
│   └── main.py               # CLI benchmark runner linking ingestion, chunking, and CPU search
├── tests/
│   └── test_geneweaver.py    # Unit and regression test suite
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore configuration
└── README.md                 # Project documentation and performance notes
```

---

## ⚡ Key Features

- **Chunked Genomic Streaming (`GenomicParser`)**: Ingests arbitrarily large FASTA genomes with minimal RAM footprint by streaming configurable bytearray slices (e.g., 1MB buffers) with boundary overlap to ensure no cut-site targets are lost.
- **In Silico Chromosome Generator (`utils.generate_mock_fasta`)**: Synthesizes customizable gigabase/megabase reference FASTA sequences with embedded on-target and off-target CRISPR sites (with canonical `NGG` PAM motifs and point mutations).
- **CPU Alignment Baseline (`baseline_cpu.cpu_align_sequence`)**: Naive sliding-window search calculating Hamming distances, execution latency, and throughput (MB/s).
- **Extensible Architecture for GPU Acceleration**: Bytearray buffers are aligned and typed for zero-copy hand-off to Numba `@cuda.jit` kernels and Dask distributed task graphs.

---

## 🚀 Quickstart

### 1. Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Day 1 CLI Benchmark

To run the end-to-end benchmark (which automatically synthesizes a 5MB FASTA if not found, parses it into 1MB chunks, and executes the CPU baseline search):

```bash
python -m geneweaver.main --fasta mock_genome_5mb.fasta --guide GTCACCGTAGTTCATAGCGG --max-mismatches 2
```

### 3. CLI Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--fasta` | `str` | `mock_genome_5mb.fasta` | Path to FASTA input file |
| `--guide` | `str` | `GTCACCGTAGTTCATAGCGG` | Guide RNA query sequence |
| `--chunk-size-mb` | `float` | `1.0` | Chunk buffer size in Megabytes |
| `--max-mismatches` | `int` | `2` | Max Hamming distance tolerated for hits |
| `--generate-size-mb` | `float` | `5.0` | Size of mock FASTA to create if missing |

---

## 🧪 Running Tests

Run the test suite using `pytest`:

```bash
pytest -v
```

---

## 🛣️ Roadmap: GPU Acceleration (Upcoming Days)

1. **Day 2**: Numba CUDA sliding-window kernel implementation with GPU shared-memory caching.
2. **Day 3**: Dask parallel chunk dispatch across multi-GPU nodes with PyNVML telemetry.
3. **Day 4**: Interactive Textual TUI dashboard for real-time throughput and memory profiling.
