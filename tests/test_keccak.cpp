/*
 * test_keccak.cpp – Unit tests for the Keccak-256 and Keccak-512 implementations.
 *
 * Verifies against known test vectors from the Keccak team and Ethereum.
 */
#include "kawpow/keccak.h"
#include "utils/hex.h"

#include <cstdio>
#include <cstring>
#include <string>

static int g_pass = 0;
static int g_fail = 0;

static void check(const char* name,
                  const std::string& got,
                  const std::string& expected) {
    if (got == expected) {
        ++g_pass;
        std::printf("  PASS  %s\n", name);
    } else {
        ++g_fail;
        std::printf("  FAIL  %s\n    got:      %s\n    expected: %s\n",
                    name, got.c_str(), expected.c_str());
    }
}

int main() {
    std::printf("=== Keccak tests ===\n");

    /* ---- Keccak-256 ---- */

    /* empty input */
    {
        auto h = kawpow::keccak256(nullptr, 0);
        auto hex = util::to_hex(h.data(), h.size());
        check("keccak256(\"\")",
              hex,
              "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470");
    }

    /* "abc" – standard Keccak test vector */
    {
        const uint8_t msg[] = {'a', 'b', 'c'};
        auto h = kawpow::keccak256(msg, 3);
        auto hex = util::to_hex(h.data(), h.size());
        check("keccak256(\"abc\")",
              hex,
              "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45");
    }

    /* 32 zero bytes (seed hash for epoch 0) */
    {
        kawpow::hash256 zeros{};
        auto h = kawpow::keccak256(zeros);
        auto hex = util::to_hex(h.data(), h.size());
        check("keccak256(32 zero bytes)",
              hex,
              "290decd9548b62a8d60345a988386fc84ba6bc95484008f6362f93160ef3e563");
    }

    /* ---- Keccak-512 ---- */

    /* empty input */
    {
        auto h = kawpow::keccak512(nullptr, 0);
        auto hex = util::to_hex(h.data(), h.size());
        check("keccak512(\"\")",
              hex,
              "0eab42de4c3ceb9235fc91acffe746b29c29a8c366b7c60e4e67c466f36a4304"
              "c00fa9caf9d87976ba469bcbe06713b435f091ef2769fb160cdab33d3670680e");
    }

    /* "abc" */
    {
        const uint8_t msg[] = {'a', 'b', 'c'};
        auto h = kawpow::keccak512(msg, 3);
        auto hex = util::to_hex(h.data(), h.size());
        check("keccak512(\"abc\")",
              hex,
              "18587dc2ea106b9a1563e32b3312421ca164c7f1f07bc922a9c83d77cea3a1e5"
              "d0c69910739025372dc14ac9642629379540c17e2a65b19d77aa511a9d00bb96");
    }

    std::printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
