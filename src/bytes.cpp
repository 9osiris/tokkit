#include "bytes.h"
#include <map>

// encode one codepoint as utf8
std::string utf8_encode(int cp) {
  std::string out;
  if (cp < 0x80) {
    out += (char)cp;
  } else if (cp < 0x800) {
    out += (char)(0xc0 | (cp >> 6));
    out += (char)(0x80 | (cp & 0x3f));
  } else if (cp < 0x10000) {
    out += (char)(0xe0 | (cp >> 12));
    out += (char)(0x80 | ((cp >> 6) & 0x3f));
    out += (char)(0x80 | (cp & 0x3f));
  } else {
    out += (char)(0xf0 | (cp >> 18));
    out += (char)(0x80 | ((cp >> 12) & 0x3f));
    out += (char)(0x80 | ((cp >> 6) & 0x3f));
    out += (char)(0x80 | (cp & 0x3f));
  }
  return out;
}

// decode utf8 to codepoints, bad bytes become u+fffd
std::vector<int> utf8_decode(const std::string& s) {
  std::vector<int> out;
  size_t i = 0;
  while (i < s.size()) {
    unsigned char c = s[i];
    int cp = 0xFFFD;
    size_t len = 1;
    if (c < 0x80) {
      cp = c;
    } else if ((c & 0xe0) == 0xc0 && i + 1 < s.size()) {
      cp = ((c & 0x1f) << 6) | (s[i + 1] & 0x3f);
      len = 2;
    } else if ((c & 0xf0) == 0xe0 && i + 2 < s.size()) {
      cp = ((c & 0x0f) << 12) | ((s[i + 1] & 0x3f) << 6) | (s[i + 2] & 0x3f);
      len = 3;
    } else if ((c & 0xf8) == 0xf0 && i + 3 < s.size()) {
      cp = ((c & 0x07) << 18) | ((s[i + 1] & 0x3f) << 12) |
           ((s[i + 2] & 0x3f) << 6) | (s[i + 3] & 0x3f);
      len = 4;
    }
    out.push_back(cp);
    i += len;
  }
  return out;
}

// the gpt2 byte table: printable bytes map to themselves, the rest
// get codepoints starting at 256 in byte order
static const int* byte_table() {
  static int table[256];
  static bool built = false;
  if (!built) {
    int bs[256];
    int cs[256];
    int n = 0;
    for (int b = 33; b <= 126; b++) bs[n++] = b;
    for (int b = 161; b <= 172; b++) bs[n++] = b;
    for (int b = 174; b <= 255; b++) bs[n++] = b;
    for (int i = 0; i < n; i++) cs[i] = bs[i];
    int m = n;
    for (int b = 0; b < 256; b++) {
      bool seen = false;
      for (int i = 0; i < n; i++) if (bs[i] == b) seen = true;
      if (!seen) {
        bs[m] = b;
        cs[m] = 256 + (m - n);
        m++;
      }
    }
    for (int i = 0; i < 256; i++) table[bs[i]] = cs[i];
    built = true;
  }
  return table;
}

int byte_to_cp(int b) {
  return byte_table()[b & 0xff];
}

int cp_to_byte(int cp) {
  const int* table = byte_table();
  for (int b = 0; b < 256; b++) if (table[b] == cp) return b;
  return -1;
}
