"""
=============================================================================
 SMOKE TEST - run the whole student experience, unattended  [ ORGANISER ]
=============================================================================
 Runs every script a participant will run, exactly as they will run it, but
 non-interactively - then prints one PASS/FAIL table.

     python smoke_test.py            # everything (~8-12 min)
     python smoke_test.py --fast     # skip the slow PyTorch demos (~3 min)
     python smoke_test.py --build    # also re-run 02 + 03 (adds ~40 min)

 Use this to validate a bundle before you hand it out. It answers the only
 question that matters on workshop morning: "does every command in the guide
 actually work on a clean machine?"

 It does NOT judge answer quality - only that each script starts, produces the
 output the guide promises, and exits cleanly.
=============================================================================
"""
import os
import re
import sys
import time
import subprocess

FAST = "--fast" in sys.argv
BUILD = "--build" in sys.argv

# The demos no longer ask anything on their own - the presenter types the
# question live. So here we type it for them: send the standard question,
# wait for the answer, then send "exit" to leave the interactive loop.
# That is exactly the participant path, automated.
STANDARD_QUESTION = (
    "I am at a remote crude-transfer pumping station. Pump P-104 has stopped "
    "repeatedly and the local controller is showing a motor overload/thermal-"
    "protection alarm. I have no internet access. What are the common causes "
    "and what should I check first?"
)
STDIN_EXIT = STANDARD_QUESTION + "\nexit\n"

RESULTS = []


def run(label, cmd, expect, timeout, feed_exit=True, optional=False):
    """
    Run one script and check its output for the markers the guide promises.

    `expect` is a list of (description, regex) pairs. Every one must match.
    """
    print(f"\n{'=' * 66}")
    print(f"RUNNING: {label}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'=' * 66}")

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            input=STDIN_EXIT if feed_exit else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        code = proc.returncode
    except subprocess.TimeoutExpired:
        RESULTS.append((label, "TIMEOUT", f"exceeded {timeout}s", 0, optional))
        print(f"  TIMEOUT after {timeout}s")
        return
    except Exception as e:
        RESULTS.append((label, "ERROR", str(e)[:60], 0, optional))
        print(f"  ERROR {e}")
        return

    elapsed = time.time() - start

    if code != 0:
        tail = "\n".join(l for l in out.strip().splitlines()[-6:])
        RESULTS.append((label, "FAIL", f"exit code {code}", elapsed, optional))
        print(f"  FAILED (exit {code}) after {elapsed:.0f}s")
        print("  --- last lines ---")
        print("  " + tail.replace("\n", "\n  "))
        return

    missing = [desc for desc, pattern in expect
               if not re.search(pattern, out, re.IGNORECASE | re.MULTILINE)]

    # Pull the measured speed out of the "Speed : 12.3s | 456 tokens | 78 t/s"
    # line specifically. A loose search for "t/s" would also match the
    # explanatory hints the demos print ("compare this with demos 1-3, about
    # 6 t/s") and report those as if they were measurements.
    speed = ""
    m = re.findall(r"Speed\s*:\s*[\d.]+s\s*\|\s*\d+\s*tokens?\s*\|\s*([\d.]+)\s*t/s",
                   out)
    if m:
        speed = f"{float(m[-1]):.1f} t/s"

    if missing:
        RESULTS.append((label, "FAIL", "missing: " + ", ".join(missing),
                        elapsed, optional))
        print(f"  RAN, but output is wrong. Missing: {', '.join(missing)}")
    else:
        RESULTS.append((label, "PASS", speed, elapsed, optional))
        print(f"  PASS in {elapsed:.0f}s  {speed}")


PY = sys.executable

# ---------------------------------------------------------------------------
# THE STUDENT PATH, IN ORDER
# ---------------------------------------------------------------------------
def main():
    print("=" * 66)
    print("WORKSHOP SMOKE TEST - running the student path unattended")
    if FAST:
        print("  --fast: skipping the slow PyTorch demos (1, 2, 3)")
    if BUILD:
        print("  --build: also re-running fine-tune + quantize (SLOW)")
    print("=" * 66)

    overall = time.time()

    # --- Setup checks -------------------------------------------------------
    run("verify_setup.py", [PY, "verify_setup.py"],
        [("result line", r"RESULT:")],
        timeout=300, feed_exit=False)

    run("inspect_system.py", [PY, "inspect_system.py"],
        [("processor section", r"PROCESSOR"),
         ("memory section", r"MEMORY")],
        timeout=300, feed_exit=False)

    # --- The one build step students run ------------------------------------
    run("01_rag_baseline.py  (build vector DB)", [PY, "01_rag_baseline.py"],
        [("chunk count", r"knowledge base ready:\s*\d+\s*chunks")],
        timeout=900, feed_exit=False)

    # --- Optional: the heavy build steps ------------------------------------
    if BUILD:
        run("02_finetune_qlora.py  (SLOW)", [PY, "02_finetune_qlora.py"],
            [("trainable params", r"trainable params:"),
             ("adapter saved", r"Adapter ready:")],
            timeout=7200, feed_exit=False)

        run("03_quantize_gguf.py", [PY, "03_quantize_gguf.py"],
            [("quantized", r"EDGE MODEL READY")],
            timeout=3600, feed_exit=False)

    # --- The five demos -----------------------------------------------------
    if not FAST:
        run("DEMO 1  base model", [PY, "tests/test_00_base_slm.py"],
            [("demo header", r"DEMO 1"),
             ("produced an answer", r"Answer:"),
             ("speed line", r"t/s")],
            timeout=1800)

        run("DEMO 2  fine-tuned", [PY, "tests/test_02_adapter.py"],
            [("demo header", r"DEMO 2"),
             ("adapter loaded", r"Adapter\s*:"),
             ("speed line", r"t/s")],
            timeout=1800)

        run("DEMO 3  fine-tuned + RAG", [PY, "tests/test_03_adapter_rag.py"],
            [("demo header", r"DEMO 3"),
             ("retrieval distance", r"Retrieval distance"),
             ("cited a source file", r"\*\s+\w+.*\.(pdf|docx|html|json)")],
            timeout=1800)
    else:
        for lbl in ("DEMO 1  base model", "DEMO 2  fine-tuned",
                    "DEMO 3  fine-tuned + RAG"):
            RESULTS.append((lbl, "SKIP", "--fast", 0, True))

    run("DEMO 4  quantized", [PY, "tests/test_04_gguf.py"],
        [("demo header", r"DEMO 4"),
         ("cpu backend", r"Backend\s*:\s*CPU"),
         ("speed line", r"t/s")],
        timeout=1200)

    run("DEMO 5  quantized + RAG", [PY, "04_gguf_rag.py"],
        [("demo header", r"DEMO 5"),
         ("cpu backend", r"Backend\s*:\s*CPU"),
         ("rag mode", r"RAG mode"),
         ("cited a source file", r"\*\s+\w+.*\.(pdf|docx|html|json)")],
        timeout=1200)

    # --- Bonus demo ---------------------------------------------------------
    if not FAST:
        run("BONUS   base + RAG", [PY, "tests/test_01_base_slm_rag.py"],
            [("demo header", r"BONUS DEMO"),
             ("retrieval distance", r"Retrieval distance")],
            timeout=1800, optional=True)

    # --- Report -------------------------------------------------------------
    total = time.time() - overall
    print("\n\n" + "=" * 66)
    print("SMOKE TEST RESULTS")
    print("=" * 66)
    print(f"{'STATUS':<9} {'SCRIPT':<38} {'TIME':>7}  DETAIL")
    print("-" * 66)
    for label, status, detail, elapsed, optional in RESULTS:
        t = f"{elapsed:.0f}s" if elapsed else "-"
        print(f"{status:<9} {label:<38} {t:>7}  {detail}")

    hard_fails = [r for r in RESULTS if r[1] not in ("PASS", "SKIP") and not r[4]]
    soft_fails = [r for r in RESULTS if r[1] not in ("PASS", "SKIP") and r[4]]

    print("-" * 66)
    print(f"Total time: {total / 60:.1f} minutes")
    print()
    if not hard_fails:
        print("RESULT: BUNDLE IS READY TO HAND OUT")
        if soft_fails:
            print(f"        ({len(soft_fails)} optional item(s) failed - not blocking)")
    else:
        print(f"RESULT: {len(hard_fails)} BLOCKING FAILURE(S)")
        for label, status, detail, _, _ in hard_fails:
            print(f"  - {label}: {status} ({detail})")
    print("=" * 66)

    return 1 if hard_fails else 0


if __name__ == "__main__":
    sys.exit(main())
