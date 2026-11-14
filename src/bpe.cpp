#include "bpe.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "bytes.h"
#include "pretoken.h"

Bpe::Bpe() { rebuild(); }

// rebuild the id maps from the merge list
void Bpe::rebuild() {
  token_to_id_.clear();
  id_to_token_.clear();
  merge_rank_.clear();
  cache_.clear();
  for (int b = 0; b < 256; b++) {
    std::string tok = utf8_encode(byte_to_cp(b));
    token_to_id_[tok] = b;
    id_to_token_.push_back(tok);
  }
  for (size_t i = 0; i < merges_.size(); i++) {
    std::string tok = merges_[i].first + merges_[i].second;
    token_to_id_[tok] = 256 + (int)i;
    id_to_token_.push_back(tok);
    merge_rank_[merges_[i]] = (int)i;
  }
}

void Bpe::train(const std::string& text, int vocab_size) {
  merges_.clear();
  int want = vocab_size - 256;
  if (want <= 0) {
    rebuild();
    return;
  }
  // count how often each pre-token shows up, in byte symbols
  std::map<std::vector<std::string>, int> words;
  for (const std::string& piece : pretokenize(text)) {
    std::vector<std::string> syms;
    for (unsigned char b : piece) syms.push_back(utf8_encode(byte_to_cp(b)));
    words[syms]++;
  }
  while ((int)merges_.size() < want) {
    // count adjacent pairs across all words, weighted by frequency
    std::map<std::pair<std::string, std::string>, int> pairs;
    for (auto& kv : words) {
      const std::vector<std::string>& w = kv.first;
      for (size_t i = 0; i + 1 < w.size(); i++) pairs[{w[i], w[i + 1]}] += kv.second;
    }
    if (pairs.empty()) break;
    // most frequent pair wins, ties break alphabetically so runs are deterministic
    auto best = pairs.begin();
    for (auto it = pairs.begin(); it != pairs.end(); it++) {
      if (it->second > best->second) best = it;
    }
    if (best->second < 2) break;  // merging a one-off pair buys nothing
    merges_.push_back(best->first);
    // rewrite every word with the new merge applied
    std::map<std::vector<std::string>, int> next;
    for (auto& kv : words) {
      std::vector<std::string> w;
      size_t i = 0;
      while (i < kv.first.size()) {
        if (i + 1 < kv.first.size() && kv.first[i] == best->first.first &&
            kv.first[i + 1] == best->first.second) {
          w.push_back(kv.first[i] + kv.first[i + 1]);
          i += 2;
        } else {
          w.push_back(kv.first[i]);
          i++;
        }
      }
      next[w] += kv.second;
    }
    words = next;
  }
  rebuild();
}

std::vector<int> Bpe::encode_piece(const std::string& piece) const {
  auto hit = cache_.find(piece);
  if (hit != cache_.end()) return hit->second;
  std::vector<std::string> syms;
  for (unsigned char b : piece) syms.push_back(utf8_encode(byte_to_cp(b)));
  // keep merging the lowest-rank pair until nothing merges
  while (syms.size() > 1) {
    int best_rank = -1;
    size_t best_i = 0;
    for (size_t i = 0; i + 1 < syms.size(); i++) {
      auto it = merge_rank_.find({syms[i], syms[i + 1]});
      if (it != merge_rank_.end() && (best_rank == -1 || it->second < best_rank)) {
        best_rank = it->second;
        best_i = i;
      }
    }
    if (best_rank == -1) break;
    std::vector<std::string> next;
    size_t i = 0;
    while (i < syms.size()) {
      if (i == best_i) {
        next.push_back(syms[i] + syms[i + 1]);
        i += 2;
      } else {
        next.push_back(syms[i]);
        i++;
      }
    }
    syms = next;
  }
  std::vector<int> ids;
  for (auto& s : syms) ids.push_back(token_to_id_.at(s));
  cache_[piece] = ids;
  return ids;
}

std::vector<int> Bpe::encode(const std::string& text) const {
  std::vector<int> out;
  for (const std::string& piece : pretokenize(text)) {
    std::vector<int> ids = encode_piece(piece);
    out.insert(out.end(), ids.begin(), ids.end());
  }
  return out;
}

std::string Bpe::decode(const std::vector<int>& ids) const {
  std::string joined;
  for (int id : ids) {
    if (id < 0 || id >= (int)id_to_token_.size())
      throw std::runtime_error("token id out of range: " + std::to_string(id));
    joined += id_to_token_[id];
  }
  std::string out;
  for (int cp : utf8_decode(joined)) {
    int b = cp_to_byte(cp);
    if (b < 0) throw std::runtime_error("token text has no byte mapping");
    out += (char)b;
  }
  return out;
}

bool Bpe::save(const std::string& dir) const {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) return false;
  std::ofstream merges(dir + "/merges.txt");
  if (!merges) return false;
  for (auto& m : merges_) merges << m.first << " " << m.second << "\n";
  std::ofstream vocab(dir + "/vocab.txt");
  if (!vocab) return false;
  for (size_t i = 0; i < id_to_token_.size(); i++)
    vocab << i << "\t" << id_to_token_[i] << "\n";
  return true;
}

bool Bpe::load(const std::string& merges_path) {
  std::ifstream in(merges_path);
  if (!in) return false;
  merges_.clear();
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    size_t sp = line.find(' ');
    if (sp == std::string::npos) return false;
    merges_.push_back({line.substr(0, sp), line.substr(sp + 1)});
  }
  rebuild();
  return true;
}

int Bpe::vocab_size() const { return (int)id_to_token_.size(); }

int Bpe::merge_count() const { return (int)merges_.size(); }

std::string Bpe::token_string(int id) const {
  if (id < 0 || id >= (int)id_to_token_.size()) return "";
  return id_to_token_[id];
}
