#pragma once
#include <map>
#include <string>
#include <vector>

// byte level bpe: vocab is the 256 byte tokens, then merges in order
class Bpe {
 public:
  Bpe();
  void train(const std::string& text, int vocab_size);
  std::vector<int> encode(const std::string& text) const;
  std::string decode(const std::vector<int>& ids) const;
  bool save(const std::string& dir) const;  // writes merges.txt and vocab.txt
  bool load(const std::string& merges_path);
  int vocab_size() const;
  int merge_count() const;
  std::string token_string(int id) const;

 private:
  std::vector<std::pair<std::string, std::string>> merges_;
  std::map<std::string, int> token_to_id_;
  std::vector<std::string> id_to_token_;
  std::map<std::pair<std::string, std::string>, int> merge_rank_;
  mutable std::map<std::string, std::vector<int>> cache_;
  void rebuild();
  std::vector<int> encode_piece(const std::string& piece) const;
};
