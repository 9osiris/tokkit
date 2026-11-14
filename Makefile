CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra

SRC = src/bytes.cpp src/pretoken.cpp src/bpe.cpp src/cli.cpp

tokkit: $(SRC)
	$(CXX) $(CXXFLAGS) -o tokkit $(SRC)

test: tokkit
	python3 tests/test_tokkit.py

clean:
	rm -f tokkit

.PHONY: test clean
