#pragma once
#include <string>
#include <vector>

// split text into pre-tokens, gpt2-ish: a space sticks to the next word
// so " hello" is one piece and roundtrips exactly
std::vector<std::string> pretokenize(const std::string& text);
