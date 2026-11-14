#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "bpe.h"

#define VERSION "1.0.0"

// read a whole file into a string, empty on failure
static std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return "";
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static std::string read_stdin() {
  std::stringstream ss;
  ss << std::cin.rdbuf();
  return ss.str();
}

static void usage() {
  std::cout << "tokkit " << VERSION << " - byte level bpe tokenizer\n"
            << "usage:\n"
            << "  tokkit train --corpus FILE --vocab-size N --out DIR\n"
            << "  tokkit encode --merges FILE [--text TEXT | --file FILE] [--ids]\n"
            << "  tokkit decode --merges FILE --ids \"12 34\" | --file FILE\n"
            << "  tokkit info --merges FILE\n"
            << "  tokkit bench --merges FILE [--text TEXT | --file FILE] [--n N]\n";
}

// grab the value after a flag, "" when missing
static std::string flag(int argc, char** argv, const std::string& name) {
  for (int i = 0; i + 1 < argc; i++)
    if (argv[i] == name) return argv[i + 1];
  return "";
}

static bool has_flag(int argc, char** argv, const std::string& name) {
  for (int i = 0; i < argc; i++)
    if (argv[i] == name) return true;
  return false;
}

static std::vector<int> parse_ids(const std::string& s) {
  std::vector<int> ids;
  std::stringstream ss(s);
  int id;
  while (ss >> id) ids.push_back(id);
  return ids;
}

int main(int argc, char** argv) {
  if (argc < 2 || has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
    usage();
    return 0;
  }
  if (has_flag(argc, argv, "--version")) {
    std::cout << "tokkit " << VERSION << "\n";
    return 0;
  }
  std::string cmd = argv[1];
  try {
    if (cmd == "train") {
      std::string corpus = flag(argc, argv, "--corpus");
      std::string out = flag(argc, argv, "--out");
      std::string vs = flag(argc, argv, "--vocab-size");
      if (corpus.empty() || out.empty() || vs.empty()) {
        std::cerr << "train needs --corpus, --vocab-size and --out\n";
        return 1;
      }
      std::string text = read_file(corpus);
      if (text.empty()) {
        std::cerr << "cannot read corpus: " << corpus << "\n";
        return 1;
      }
      Bpe bpe;
      bpe.train(text, std::stoi(vs));
      if (!bpe.save(out)) {
        std::cerr << "cannot write to: " << out << "\n";
        return 1;
      }
      std::cout << "trained " << bpe.merge_count() << " merges, vocab size "
                << bpe.vocab_size() << "\n";
      std::cout << "wrote " << out << "/merges.txt and " << out << "/vocab.txt\n";
      return 0;
    }
    if (cmd == "encode" || cmd == "decode" || cmd == "info" || cmd == "bench") {
      std::string merges = flag(argc, argv, "--merges");
      if (merges.empty()) {
        std::cerr << cmd << " needs --merges FILE\n";
        return 1;
      }
      Bpe bpe;
      if (!bpe.load(merges)) {
        std::cerr << "cannot read merges: " << merges << "\n";
        return 1;
      }
      if (cmd == "info") {
        std::cout << "vocab size: " << bpe.vocab_size() << "\n";
        std::cout << "merges: " << bpe.merge_count() << "\n";
        std::cout << "first merges:\n";
        for (int i = 0; i < 10 && i < bpe.merge_count(); i++)
          std::cout << "  " << bpe.token_string(256 + i) << "\n";
        return 0;
      }
      std::string text;
      std::string file = flag(argc, argv, "--file");
      if (!file.empty()) {
        text = read_file(file);
        if (text.empty() && !std::ifstream(file)) {
          std::cerr << "cannot read file: " << file << "\n";
          return 1;
        }
      } else if (has_flag(argc, argv, "--text")) {
        text = flag(argc, argv, "--text");
      } else if (cmd == "decode" && has_flag(argc, argv, "--ids")) {
        // --ids is a value for decode, a plain flag for encode
        text = flag(argc, argv, "--ids");
      } else {
        text = read_stdin();
      }
      if (cmd == "encode") {
        std::vector<int> ids = bpe.encode(text);
        if (has_flag(argc, argv, "--ids")) {
          for (size_t i = 0; i < ids.size(); i++) {
            if (i) std::cout << " ";
            std::cout << ids[i];
          }
          std::cout << "\n";
        } else {
          for (size_t i = 0; i < ids.size(); i++) {
            if (i) std::cout << " ";
            std::cout << bpe.token_string(ids[i]);
          }
          std::cout << "\n";
        }
        std::cerr << ids.size() << " tokens\n";
        return 0;
      }
      if (cmd == "decode") {
        std::cout << bpe.decode(parse_ids(text));
        return 0;
      }
      // bench
      int n = 200;
      std::string ns = flag(argc, argv, "--n");
      if (!ns.empty()) n = std::stoi(ns);
      auto start = std::chrono::steady_clock::now();
      size_t total = 0;
      for (int i = 0; i < n; i++) total += bpe.encode(text).size();
      auto end = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(end - start).count();
      std::cout << total / n << " tokens, " << n << " encodes in " << ms << " ms ("
                << (total / (ms / 1000.0)) << " tok/s)\n";
      return 0;
    }
    std::cerr << "unknown command: " << cmd << "\n";
    usage();
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
