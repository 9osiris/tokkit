#pragma once
#include <string>
#include <vector>

// byte level alphabet, gpt2 style: every byte maps to a printable codepoint
// so merged tokens never produce invalid utf8

std::string utf8_encode(int cp);
std::vector<int> utf8_decode(const std::string& s);
int byte_to_cp(int b);
int cp_to_byte(int cp);
