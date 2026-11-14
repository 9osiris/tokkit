tokkit - byte level bpe tokenizer in c++ [educational project]

disclaimer:
this is a learning project, not a production tokenizer. it trains real
byte level bpe and roundtrips text exactly, but the pre-tokenizer is a
hand rolled scanner, training is single threaded, and there are no
special tokens. use it to understand how tokenizers work, not to ship
one.

why i made this
i keep building stuff around models (mule, probe, kvblocks) and every
one of them treats tokenization as someone else's problem. i wanted to
know what actually happens between raw text and token ids, so i wrote
one from scratch. byte level bpe because that is what gpt2 does: every
byte maps to a printable codepoint, merges are learned from pair
counts, and encode is just applying the merges in rank order. i
avoided pulling in a regex engine or a tokenizer library on purpose,
the whole point was to see the moving parts. the pre-tokenizer is a
small scanner instead of a regex, which means contractions like "don't"
split into "'" + "don't" pieces instead of getting their own rule. that
is a real gap and it is documented in the code.

the guide
build it:
  make

train a vocab on the sample corpus:
  ./tokkit train --corpus examples/corpus.txt --vocab-size 400 --out /tmp/vocab

encode some text (token text by default, --ids for numbers):
  ./tokkit encode --merges /tmp/vocab/merges.txt --text "the agent reads the prompt"
  ./tokkit encode --merges /tmp/vocab/merges.txt --text "the agent reads the prompt" --ids

decode ids back to text:
  ./tokkit decode --merges /tmp/vocab/merges.txt --ids "290 288 365 258 326"

look at a vocab, or time the encoder:
  ./tokkit info --merges /tmp/vocab/merges.txt
  ./tokkit bench --merges /tmp/vocab/merges.txt --text "the agent reads the prompt"

encode reads --text, --file, or stdin when neither is given. decode
reads --ids or --file. a tiny pre-trained vocab (320 tokens) ships in
examples/tiny if you want to try encode without training first.

run the tests:
  make test
  33 checks, builds the binary and runs it end to end. no api key, no
  network, no dependencies beyond g++ and python3.

how it works
  src/bytes.cpp - utf8 helpers and the gpt2 byte alphabet (bytes 0-255
    map to printable codepoints, space becomes Ġ)
  src/pretoken.cpp - splits text into pieces, a space sticks to the
    next word so " hello" is one piece
  src/bpe.cpp - train counts adjacent pairs and merges the most
    frequent one until the vocab size is hit (pairs seen once are
    skipped, they buy nothing), encode applies merges lowest rank
    first, decode joins tokens back to bytes
  src/cli.cpp - train, encode, decode, info, bench

merges.txt is one merge per line ("left right" in encoder chars) and
vocab.txt is "id <tab> token". vocab is the 256 byte tokens first,
then merges in order, so ids are stable for a given merges file.
