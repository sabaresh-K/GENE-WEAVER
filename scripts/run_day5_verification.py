"""Day 5 Verification and QA test runner for GeneWeaver Phase 1.

Executes:
1. Complete pytest test suite (unit and integration tests).
2. End-to-end smoke pipeline:
   - Mock FASTA dataset generation with ground truth CRISPR targets.
   - GenomicParser streaming chunking (with sequence overlap).
   - CPUAligner sliding-window uint8 alignment.
   - Verification of ground-truth target recovery.
3. Baseline report validation and summary printing.

Exits with code 0 on complete verification success, or code 1 on failure.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from geneweaver.aligner_cpu import CPUAligner
from geneweaver.baseline_report import print_week1_baseline_report
from geneweaver.parser import GenomicParser
from geneweaver.utils import generate_mock_fasta


def run_command_live(cmd: list[str], description: str) -> bool:
    """Execute a subprocess command, stream output, and return success status."""
    print(f"\n[+] Running: {description}...")
    print(f"    Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode != 0:
        print(f"[-] FAILED: {description} (exit code {result.returncode})", file=sys.stderr)
        return False
    print(f"[OK] PASSED: {description}")
    return True


def run_e2e_smoke_test() -> bool:
    """Run an end-to-end integration smoke test from mock generation to target detection."""
    print("\n[+] Running Step 2: End-to-End Pipeline Smoke Test...")
    smoke_fasta = project_root / "smoke_test_genome.fasta"
    guide = "ATCGGCGATCGATTACGGTA"  # 20nt SpCas9 prototype

    try:
        # 1. Generate small 100KB mock genome with 5 injected targets (exact and 1-2 mm)
        t0 = time.perf_counter()
        _, injected_targets = generate_mock_fasta(
            filename=smoke_fasta,
            num_megabases=0.1,  # 100,000 bases
            guide_rnas=[guide],
            num_targets=5,
            mismatch_distribution={0: 0.4, 1: 0.4, 2: 0.2},
            default_pam="CGG",
        )
        print(f"    -> Generated {len(injected_targets)} ground-truth targets in {time.perf_counter() - t0:.2f}s")

        # 2. Parse and chunk with overlap
        chunk_size = 25_000
        overlap = 50
        parser = GenomicParser(filepath=smoke_fasta)
        aligner = CPUAligner(guide_rna=guide, max_mismatches=2, check_pam=True)

        recovered_loci: list[int] = []
        for rec_id, offset, chunk_buf in parser.stream_chunks(
            chunk_size=chunk_size, overlap=overlap, return_format="numpy"
        ):
            res = aligner.align_chunk(chunk_buf)
            for hit in res.hits:
                global_pos = offset + hit.offset
                recovered_loci.append(global_pos)

        # 3. Verify that all injected targets within max_mismatches=2 were identified
        eligible_targets = [t for t in injected_targets if t.mismatches <= 2]
        all_found = True
        for target in eligible_targets:
            # Check if target.start_pos was detected
            if target.start_pos not in recovered_loci:
                print(f"    [-] Missed target at start pos {target.start_pos} (mismatches={target.mismatches})")
                all_found = False

        if not all_found:
            print("[-] Smoke test failed: Some eligible injected targets were missed.", file=sys.stderr)
            return False

        print(f"    -> All {len(eligible_targets)} ground-truth targets verified and recovered across chunk boundaries!")
        print("[OK] PASSED: End-to-End Pipeline Smoke Test")
        return True

    finally:
        # Cleanup temporary smoke test file
        if smoke_fasta.is_file():
            smoke_fasta.unlink(missing_ok=True)


def main() -> int:
    """Execute complete Day 5 verification."""
    print("=" * 78)
    print("      GeneWeaver: Day 5 Phase 1 / Week 1 Verification Suite")
    print("=" * 78)

    # 1. Run pytest suite
    python_exe = sys.executable
    pytest_ok = run_command_live(
        [python_exe, "-m", "pytest", "-v"],
        description="Full Pytest Unit & Integration Test Suite",
    )
    if not pytest_ok:
        return 1

    # 2. Run E2E smoke test
    smoke_ok = run_e2e_smoke_test()
    if not smoke_ok:
        return 1

    # 3. Print baseline report
    print("\n[+] Running Step 3: Validating & Displaying Week 1 Baseline Report...")
    baseline_artifact = project_root / "benchmarks" / "cpu_baseline.json"
    if not baseline_artifact.is_file():
        print(f"[-] Missing baseline artifact: {baseline_artifact}", file=sys.stderr)
        return 1

    try:
        print_week1_baseline_report(baseline_artifact, target_gpu_multiplier=100.0)
    except Exception as e:
        print(f"[-] Failed to generate baseline report: {e}", file=sys.stderr)
        return 1

    print("\n[OK] ALL PHASE 1 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("Ready to begin Phase 2 (Numba CUDA Kernel Compilation & Acceleration) on Day 6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
