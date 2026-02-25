/*
 * test_ethash.cpp – Unit tests for seed hash and light-cache generation.
 */
#include "kawpow/ethash.h"
#include "kawpow/keccak.h"
#include "utils/hex.h"

#include <cstdio>
#include <cstring>

static int g_pass = 0;
static int g_fail = 0;

static void check(const char* name, bool cond) {
    if (cond) {
        ++g_pass;
        std::printf("  PASS  %s\n", name);
    } else {
        ++g_fail;
        std::printf("  FAIL  %s\n", name);
    }
}

static void check_hex(const char* name,
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
    std::printf("=== Ethash tests ===\n");

    /* epoch_from_block (Ravencoin: epoch_length = 7500) */
    check("epoch_from_block(0) == 0",       kawpow::epoch_from_block(0) == 0);
    check("epoch_from_block(7499) == 0",     kawpow::epoch_from_block(7499) == 0);
    check("epoch_from_block(7500) == 1",     kawpow::epoch_from_block(7500) == 1);
    check("epoch_from_block(15000) == 2",    kawpow::epoch_from_block(15000) == 2);

    /* seed_hash: epoch 0 should be 32 zero bytes */
    {
        auto s = kawpow::seed_hash(0);
        kawpow::hash256 zeros{};
        check("seed_hash(0) == 32 zero bytes", s == zeros);
    }

    /* seed_hash: epoch 1 = keccak256(zeros) */
    {
        kawpow::hash256 zeros{};
        auto expected = kawpow::keccak256(zeros);
        auto s = kawpow::seed_hash(1);
        auto hex_got = util::to_hex(s.data(), s.size());
        auto hex_exp = util::to_hex(expected.data(), expected.size());
        check_hex("seed_hash(1) == keccak256(zeros)", hex_got, hex_exp);
    }

    /* light_cache_size: epoch 0 should be 16 MiB region (the initial size) */
    {
        uint64_t sz = kawpow::light_cache_size(0);
        check("light_cache_size(0) > 0", sz > 0);
        check("light_cache_size(0) % 64 == 0", sz % 64 == 0);
        /* should be close to 2^24 */
        check("light_cache_size(0) <= 2^24", sz <= (1ULL << 24));
        check("light_cache_size(0) > 2^24 - 1024", sz > (1ULL << 24) - 1024);
    }

    /* full_dataset_size: epoch 0 should be ~1 GiB */
    {
        uint64_t sz = kawpow::full_dataset_size(0);
        check("full_dataset_size(0) > 0", sz > 0);
        check("full_dataset_size(0) % 256 == 0", sz % 256 == 0);
        check("full_dataset_size(0) near 1 GiB",
              sz > (1ULL << 30) - 1024 && sz <= (1ULL << 30));
    }

    /* make_light_cache: basic smoke test */
    {
        auto lc = kawpow::make_light_cache(0);
        check("make_light_cache(0).num_items > 0", lc.num_items > 0);
        check("make_light_cache(0).data.size() == num_items",
              lc.data.size() == lc.num_items);

        /* first item should be deterministic */
        auto hex = util::to_hex(lc.data[0].data(), 8);
        check("first cache item is non-zero", hex != "0000000000000000");
    }

    /* calc_dag_item: basic sanity */
    {
        auto lc = kawpow::make_light_cache(0);
        auto item0 = kawpow::calc_dag_item(lc, 0);
        auto item1 = kawpow::calc_dag_item(lc, 1);
        /* items should differ */
        check("dag_item(0) != dag_item(1)", item0 != item1);
        /* item should be non-zero */
        kawpow::hash512 zeros{};
        check("dag_item(0) != zeros", item0 != zeros);
    }

    std::printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
