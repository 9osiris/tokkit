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

        # special tokens: atomic ids after the regular vocab
        spdir = os.path.join(tmp, "special")
        r = run([BIN, "train", "--corpus", CORPUS, "--vocab-size", "400",
                 "--out", spdir, "--specials", "<|endoftext|>,<|fim_prefix|>"])
        check("train with specials exits 0", r.returncode == 0)
        spfile = os.path.join(spdir, "specials.txt")
        check("specials.txt exists",
              os.path.exists(spfile) and
              open(spfile).read().splitlines() == ["<|endoftext|>", "<|fim_prefix|>"])
        spmerges = os.path.join(spdir, "merges.txt")
        nmerges = len(open(spmerges).read().splitlines())
        spvocab = open(os.path.join(spdir, "vocab.txt")).read().splitlines()
        check("vocab grows by the specials", len(spvocab) == 256 + nmerges + 2)
        eot = 256 + nmerges
        r = run([BIN, "encode", "--merges", spmerges, "--text",
                 "the agent<|endoftext|>reads", "--ids"])
        check("special token encodes to its own id",
              r.returncode == 0 and str(eot) in r.stdout.split())
        check("special id is used exactly once",
              r.stdout.split().count(str(eot)) == 1)
        ids = r.stdout.strip()
        r2 = run([BIN, "decode", "--merges", spmerges, "--ids", ids])
        check("roundtrip with specials",
              r2.returncode == 0 and r2.stdout == "the agent<|endoftext|>reads")
        r = run([BIN, "encode", "--merges", spmerges, "--text",
                 "<|endoftext|>the agent<|fim_prefix|>", "--ids"])
        check("specials at the edges encode",
              r.returncode == 0 and str(eot) in r.stdout.split() and
              str(eot + 1) in r.stdout.split())
        r2 = run([BIN, "decode", "--merges", spmerges, "--ids", r.stdout.strip()])
        check("roundtrip with specials at the edges",
              r2.returncode == 0 and r2.stdout == "<|endoftext|>the agent<|fim_prefix|>")
        r = run([BIN, "encode", "--merges", spmerges, "--text",
                 "endoftext is just a word here", "--ids"])
        check("bare text never becomes a special",
              r.returncode == 0 and str(eot) not in r.stdout.split())
        r = run([BIN, "decode", "--merges", spmerges, "--ids", str(eot)])
        check("decoding a special id gives its text",
              r.returncode == 0 and r.stdout == "<|endoftext|>")
        r = run([BIN, "info", "--merges", spmerges])
        check("info lists specials",
              r.returncode == 0 and "specials: 2" in r.stdout and
              "<|endoftext|>" in r.stdout)
        tiny_merges = os.path.join(ROOT, "examples", "tiny", "merges.txt")
        r = run([BIN, "encode", "--merges", tiny_merges, "--text",
                 "the<|endoftext|>agent", "--ids"])
        check("shipped tiny vocab has endoftext",
              r.returncode == 0 and "320" in r.stdout.split())
        r2 = run([BIN, "decode", "--merges", tiny_merges, "--ids", r.stdout.strip()])
        check("tiny vocab roundtrips a special",
              r2.returncode == 0 and r2.stdout == "the<|endoftext|>agent")
        r = run([BIN, "train", "--corpus", CORPUS, "--vocab-size", "400",
                 "--out", os.path.join(tmp, "dup"),
                 "--specials", "<|endoftext|>,<|endoftext|>"])
        check("duplicate special fails", r.returncode != 0)
        r = run([BIN, "train", "--corpus", CORPUS, "--vocab-size", "400",
                 "--out", os.path.join(tmp, "sp"),
                 "--specials", "has a space"])
        check("special with a space fails", r.returncode != 0)

        # bench runs and reports timing
        r = run([BIN, "bench", "--merges", merges, "--text", samples[0], "--n", "20"])
        check("bench prints tok/s", r.returncode == 0 and "tok/s" in r.stdout)

        # unknown command fails
        r = run([BIN, "frobnicate"])
        check("unknown command fails", r.returncode != 0)

    print(f"\n{passed} passed")


if __name__ == "__main__":
    main()
