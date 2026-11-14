#include "pretoken.h"

// classification works on raw bytes: non-ascii bytes join word runs,
// so accented words like café stay in one piece
static bool is_letter(unsigned char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80;
}

static bool is_digit(unsigned char c) { return c >= '0' && c <= '9'; }

static bool is_space(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

std::vector<std::string> pretokenize(const std::string& text) {
  std::vector<std::string> out;
  size_t i = 0;
  size_t n = text.size();
  while (i < n) {
    unsigned char c = (unsigned char)text[i];
    size_t j = i + 1;
    if (c == ' ') {
      // space attaches to the run after it, or to following whitespace
      if (j < n && is_letter((unsigned char)text[j])) {
        while (j < n && is_letter((unsigned char)text[j])) j++;
      } else if (j < n && is_digit((unsigned char)text[j])) {
        while (j < n && is_digit((unsigned char)text[j])) j++;
      } else if (j < n && !is_space((unsigned char)text[j])) {
        while (j < n && !is_space((unsigned char)text[j]) &&
               !is_letter((unsigned char)text[j]) &&
               !is_digit((unsigned char)text[j]))
          j++;
      } else {
        while (j < n && is_space((unsigned char)text[j])) j++;
      }
    } else if (is_letter(c)) {
      while (j < n && is_letter((unsigned char)text[j])) j++;
    } else if (is_digit(c)) {
      while (j < n && is_digit((unsigned char)text[j])) j++;
    } else if (is_space(c)) {
      while (j < n && is_space((unsigned char)text[j])) j++;
    } else {
      // punctuation and symbols run together until a letter, digit or space
      while (j < n && !is_space((unsigned char)text[j]) &&
             !is_letter((unsigned char)text[j]) &&
             !is_digit((unsigned char)text[j]))
        j++;
    }
    out.push_back(text.substr(i, j - i));
    i = j;
  }
  return out;
}
