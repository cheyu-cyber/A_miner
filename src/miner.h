/*
 * miner.h – Main miner orchestration for KawPow (Ravencoin).
 *
 * Coordinates the stratum client, DAG management, and mining loops
 * (CPU reference or CUDA GPU).
 */
#pragma once

#include "stratum/stratum_client.h"
#include "kawpow/ethash.h"
#include "kawpow/kawpow.h"

#include <cstdint>
#include <string>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <chrono>

class Miner {
public:
    struct Config {
        std::string pool_host;
        uint16_t    pool_port  = 0;
        std::string wallet;        /* RVN address or user.worker */
        std::string worker_pass  = "x";
        int         gpu_id       = 0;
        bool        cpu_verify   = false;  /* run CPU verifier in parallel */
    };

    explicit Miner(const Config& cfg);
    ~Miner();

    /* Start / stop mining */
    void start();
    void stop();

    /* Statistics */
    double hashrate() const;       /* hashes / sec */
    uint64_t shares_found() const;

private:
    void stratum_thread();
    void mining_thread();
    void on_new_job(const stratum::Job& job);

    Config                  m_cfg;
    stratum::StratumClient  m_stratum;

    /* current job (protected by m_job_mutex) */
    std::mutex              m_job_mutex;
    stratum::Job            m_current_job;
    bool                    m_has_job = false;

    /* DAG / cache */
    uint32_t                m_current_epoch = UINT32_MAX;
    kawpow::LightCache      m_light_cache;

    /* threads */
    std::thread             m_stratum_th;
    std::thread             m_mining_th;
    std::atomic<bool>       m_running{false};

    /* stats */
    std::atomic<uint64_t>   m_hash_count{0};
    std::atomic<uint64_t>   m_shares{0};
    std::chrono::steady_clock::time_point m_start_time;
    mutable std::mutex      m_stats_mutex;
    mutable uint64_t        m_last_hash_count = 0;
    mutable std::chrono::steady_clock::time_point m_last_hashrate_time;
};
