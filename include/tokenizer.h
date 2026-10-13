#pragma once

#include <string>
#include <vector>
#include <unordered_map>

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer() = default;

    bool load(const std::string& binary_path);
    void load_from_vocab(const std::vector<std::string>& vocab, const std::vector<float>& scores);

    std::vector<int> encode(const std::string& text, bool bos = true, bool eos = false) const;
    std::string decode(int token_id) const;
    std::string decode(const std::vector<int>& tokens) const;

    int vocab_size() const;
    int bos_id() const;
    int eos_id() const;

private:
    std::vector<std::string> id_to_token_;
    std::vector<float> id_to_score_;
    std::unordered_map<std::string, int> token_to_id_;
    int bos_id_;
    int eos_id_;
    int max_token_length_;
};
