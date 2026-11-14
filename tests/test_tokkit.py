#!/usr/bin/env python3
# build tokkit and check it end to end, no api key, no network
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "tokkit")
CORPUS = os.path.join(ROOT, "examples", "corpus.txt")

passed = 0


def check(name, cond):
    global passed
    if not cond:
        print(f"FAIL: {name}")
        sys.exit(1)
    passed += 1
    print(f"ok: {name}")


def run(args, inp=None):
    return subprocess.run(args, input=inp, capture_output=True, text=True)


def main():
    r = subprocess.run(["make", "-C", ROOT], capture_output=True, text=True)
    check("builds clean", r.returncode == 0 and os.path.exists(BIN))

    with tempfile.TemporaryDirectory() as tmp:
        outdir = os.path.join(tmp, "vocab")
        r = run([BIN, "train", "--corpus", CORPUS, "--vocab-size", "400",
                 "--out", outdir])
        check("train exits 0", r.returncode == 0)
        merges = os.path.join(outdir, "merges.txt")
        vocab = os.path.join(outdir, "vocab.txt")
        check("merges.txt exists", os.path.exists(merges))
        check("vocab.txt exists", os.path.exists(vocab))
        mlines = open(merges).read().splitlines()
        vlines = open(vocab).read().splitlines()
        check("vocab has 256 + merges lines", len(vlines) == 256 + len(mlines))
        check("learned a real number of merges", len(mlines) > 50)
        check("vocab capped at 400", len(vlines) <= 400)

        samples = [
            "the agent reads the prompt and decides what to do next.",
            "hello world, this is a test 123.",
            "the café on the corner serves a naïve blend.",
            "tokens: 1, 2, 3... and $40.50?!",
            "line one\nline two\n\nline four",
            "   leading spaces and trailing spaces   ",
            "",
        ]
        for s in samples:
            r = run([BIN, "encode", "--merges", merges, "--text", s, "--ids"])
            check(f"encode exits 0 for {s[:20]!r}", r.returncode == 0)
            ids = r.stdout.strip()
            r2 = run([BIN, "decode", "--merges", merges, "--ids", ids])
            check(f"roundtrip {s[:20]!r}", r2.returncode == 0 and r2.stdout == s)

        # determinism: same input, same ids, twice
        a = run([BIN, "encode", "--merges", merges, "--text", samples[0], "--ids"])
        b = run([BIN, "encode", "--merges", merges, "--text", samples[0], "--ids"])
        check("encode is deterministic", a.stdout == b.stdout)

        # frequent words collapse: " the" should be few tokens after training
        r = run([BIN, "encode", "--merges", merges, "--text", "the agent", "--ids"])
        check("'the agent' compresses to few tokens",
              r.returncode == 0 and len(r.stdout.split()) <= 4)

        # encode from stdin and from a file
        r = run([BIN, "encode", "--merges", merges, "--ids"], inp="hello world")
        check("encode reads stdin", r.returncode == 0 and r.stdout.strip() != "")
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("the cache saves money")
            fname = f.name
        r = run([BIN, "encode", "--merges", merges, "--file", fname, "--ids"])
        check("encode reads --file", r.returncode == 0 and r.stdout.strip() != "")
        os.unlink(fname)

        # token text mode prints readable tokens
        r = run([BIN, "encode", "--merges", merges, "--text", "the agent reads"])
        check("token text mode prints tokens",
              r.returncode == 0 and "agent" in r.stdout and "reads" in r.stdout)

        # info prints vocab size and merges
        r = run([BIN, "info", "--merges", merges])
        check("info shows vocab size", r.returncode == 0 and
              f"vocab size: {len(vlines)}" in r.stdout)
        check("info shows merge count", f"merges: {len(mlines)}" in r.stdout)

        # decode rejects an out-of-range id
        r = run([BIN, "decode", "--merges", merges, "--ids", "99999"])
        check("decode rejects bad id", r.returncode != 0)

        # tiny vocab trains only a few merges
        tiny = os.path.join(tmp, "tiny")
        r = run([BIN, "train", "--corpus", CORPUS, "--vocab-size", "260", "--out", tiny])
        check("tiny vocab trains 4 merges",
              r.returncode == 0 and len(open(os.path.join(tiny, "merges.txt")).read().splitlines()) == 4)

        # missing merges file fails cleanly
        r = run([BIN, "encode", "--merges", os.path.join(tmp, "nope.txt"), "--text", "hi"])
        check("missing merges fails cleanly", r.returncode != 0)

        # bench runs and reports timing
        r = run([BIN, "bench", "--merges", merges, "--text", samples[0], "--n", "20"])
        check("bench prints tok/s", r.returncode == 0 and "tok/s" in r.stdout)

        # unknown command fails
        r = run([BIN, "frobnicate"])
        check("unknown command fails", r.returncode != 0)

    print(f"\n{passed} passed")


if __name__ == "__main__":
    main()
