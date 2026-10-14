#include "tokenizer.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cstdint>

Tokenizer::Tokenizer() : bos_id_(1), eos_id_(2), max_token_length_(0) {}

bool Tokenizer::load(const std::string& binary_path) {
    std::ifstream file(binary_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    uint32_t vocab_size = 0;
    uint32_t max_len = 0;
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&max_len), sizeof(uint32_t));

    if (!file.good() || vocab_size == 0) {
        return false;
    }

    max_token_length_ = max_len;
    id_to_token_.clear();
    id_to_score_.clear();
    token_to_id_.clear();

    id_to_token_.resize(vocab_size);
    id_to_score_.resize(vocab_size);

    for (uint32_t i = 0; i < vocab_size; i++) {
        float score = 0.0f;
        int32_t len = 0;
        file.read(reinterpret_cast<char*>(&score), sizeof(float));
        file.read(reinterpret_cast<char*>(&len), sizeof(int32_t));

        std::string piece(len, '\0');
        file.read(&piece[0], len);

        id_to_token_[i] = piece;
        id_to_score_[i] = score;
        token_to_id_[piece] = static_cast<int>(i);
    }

    auto it_bos = token_to_id_.find("<s>");
    if (it_bos != token_to_id_.end()) {
        bos_id_ = it_bos->second;
    }

    auto it_eos = token_to_id_.find("</s>");
    if (it_eos != token_to_id_.end()) {
        eos_id_ = it_eos->second;
    }

    return true;
}

void Tokenizer::load_from_vocab(const std::vector<std::string>& vocab, const std::vector<float>& scores) {
    id_to_token_ = vocab;
    id_to_score_ = scores;
    token_to_id_.clear();
    max_token_length_ = 0;

    for (size_t i = 0; i < vocab.size(); i++) {
        token_to_id_[vocab[i]] = static_cast<int>(i);
        if (static_cast<int>(vocab[i].length()) > max_token_length_) {
            max_token_length_ = static_cast<int>(vocab[i].length());
        }
    }

    auto it_bos = token_to_id_.find("<s>");
    if (it_bos != token_to_id_.end()) {
        bos_id_ = it_bos->second;
    } else {
        bos_id_ = 1;
    }

    auto it_eos = token_to_id_.find("</s>");
    if (it_eos != token_to_id_.end()) {
        eos_id_ = it_eos->second;
    } else {
        eos_id_ = 2;
    }
}

std::vector<int> Tokenizer::encode(const std::string& text, bool bos, bool eos) const {
    std::vector<int> tokens;
    if (text.empty()) {
        if (bos) tokens.push_back(bos_id_);
        if (eos) tokens.push_back(eos_id_);
        return tokens;
    }

    if (bos) {
        tokens.push_back(bos_id_);
    }

    std::string processed_text;
    std::string sp_space = "\xe2\x96\x81";
    processed_text += sp_space;
    for (char c : text) {
        if (c == ' ') {
            processed_text += sp_space;
        } else {
            processed_text += c;
        }
    }

    std::vector<int> char_tokens;
    for (size_t i = 0; i < processed_text.size(); ) {
        bool found = false;
        for (size_t len = std::min(processed_text.size() - i, (size_t)8); len > 0; len--) {
            std::string sub = processed_text.substr(i, len);
            auto it = token_to_id_.find(sub);
            if (it != token_to_id_.end()) {
                char_tokens.push_back(it->second);
                i += len;
                found = true;
                break;
            }
        }
        if (!found) {
            uint8_t byte_val = static_cast<uint8_t>(processed_text[i]);
            char byte_buf[16];
            snprintf(byte_buf, sizeof(byte_buf), "<0x%02X>", byte_val);
            auto it = token_to_id_.find(byte_buf);
            if (it != token_to_id_.end()) {
                char_tokens.push_back(it->second);
            } else {
                // Fallback to <unk> token (id 0) instead of silently dropping
                auto unk_it = token_to_id_.find("<unk>");
                if (unk_it != token_to_id_.end()) {
                    char_tokens.push_back(unk_it->second);
                } else {
                    char_tokens.push_back(0);
                }
            }
            i++;
        }
    }

    while (char_tokens.size() >= 2) {
        float best_score = -1e9f;
        int best_idx = -1;
        int best_merged_id = -1;

        for (size_t i = 0; i < char_tokens.size() - 1; i++) {
            std::string merged = id_to_token_[char_tokens[i]] + id_to_token_[char_tokens[i + 1]];
            auto it = token_to_id_.find(merged);
            if (it != token_to_id_.end()) {
                float score = id_to_score_[it->second];
                if (score > best_score) {
                    best_score = score;
                    best_idx = static_cast<int>(i);
                    best_merged_id = it->second;
                }
            }
        }

        if (best_idx == -1) {
            break;
        }

        char_tokens[best_idx] = best_merged_id;
        char_tokens.erase(char_tokens.begin() + best_idx + 1);
    }

    tokens.insert(tokens.end(), char_tokens.begin(), char_tokens.end());

    if (eos) {
        tokens.push_back(eos_id_);
    }

    return tokens;
}

std::string Tokenizer::decode(int token_id) const {
    if (token_id < 0 || token_id >= static_cast<int>(id_to_token_.size())) {
        return "";
    }

    if (token_id == bos_id_ || token_id == eos_id_) {
        return "";
    }

    const std::string& piece = id_to_token_[token_id];

    if (piece.rfind("<0x", 0) == 0 && piece.back() == '>' && piece.length() == 6) {
        int byte_val = 0;
        if (sscanf(piece.c_str(), "<0x%02X>", &byte_val) == 1) {
            return std::string(1, static_cast<char>(byte_val));
        }
    }

    std::string result;
    std::string sp_space = "\xe2\x96\x81";
    size_t pos = 0;
    while (pos < piece.length()) {
        if (piece.substr(pos, sp_space.length()) == sp_space) {
            result += ' ';
            pos += sp_space.length();
        } else {
            result += piece[pos];
            pos++;
        }
    }

    return result;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int t : tokens) {
        result += decode(t);
    }
    return result;
}

int Tokenizer::vocab_size() const {
    return static_cast<int>(id_to_token_.size());
}

int Tokenizer::bos_id() const {
    return bos_id_;
}

int Tokenizer::eos_id() const {
    return eos_id_;
}
