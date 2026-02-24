/*
 * miner.cpp – Main miner orchestration.
 */
#include "miner.h"
#include "utils/log.h"
#include "utils/hex.h"

#include <chrono>
#include <cstring>
#include <algorithm>

#ifdef HAVE_CUDA
extern void cuda_mine(
    int gpu_id,
    const kawpow::hash256& header_hash,
    uint64_t start_nonce,
    uint64_t block_number,
    const kawpow::hash256& target,
    const kawpow::LightCache& cache,
    std::atomic<bool>& running,
    std::atomic<uint64_t>& hash_count,
    std::function<void(uint64_t nonce, const kawpow::KawpowResult& result)> on_solution);
#endif

Miner::Miner(const Config& cfg) : m_cfg(cfg) {}

Miner::~Miner() {
    stop();
}

void Miner::start() {
    if (m_running) return;
    m_running = true;
    m_hash_count = 0;
    m_shares = 0;
    m_start_time = std::chrono::steady_clock::now();

    /* configure stratum */
    m_stratum.set_pool(m_cfg.pool_host, m_cfg.pool_port);
    m_stratum.set_credentials(m_cfg.wallet, m_cfg.worker_pass);
    m_stratum.set_on_job([this](const stratum::Job& j) { on_new_job(j); });

    m_stratum_th = std::thread(&Miner::stratum_thread, this);
    m_mining_th  = std::thread(&Miner::mining_thread, this);

    util::log_info("Miner started");
}

void Miner::stop() {
    m_running = false;
    m_stratum.disconnect();
    if (m_stratum_th.joinable()) m_stratum_th.join();
    if (m_mining_th.joinable())  m_mining_th.join();
    util::log_info("Miner stopped");
}

double Miner::hashrate() const {
    auto elapsed = std::chrono::steady_clock::now() - m_start_time;
    double secs = std::chrono::duration<double>(elapsed).count();
    return (secs > 0) ? m_hash_count.load() / secs : 0.0;
}

uint64_t Miner::shares_found() const {
    return m_shares.load();
}

void Miner::on_new_job(const stratum::Job& job) {
    std::lock_guard<std::mutex> lock(m_job_mutex);
    m_current_job = job;
    m_has_job = true;

    /* rebuild cache if epoch changed */
    uint32_t epoch = kawpow::epoch_from_block(job.block_number);
    if (epoch != m_current_epoch) {
        util::log_info("Building light cache for epoch %u ...", epoch);
        m_light_cache    = kawpow::make_light_cache(epoch);
        m_current_epoch  = epoch;
        util::log_info("Light cache ready (%u items)", m_light_cache.num_items);
    }
}

void Miner::stratum_thread() {
    while (m_running) {
        if (!m_stratum.is_connected()) {
            util::log_info("Connecting to pool %s:%u ...",
                           m_cfg.pool_host.c_str(), m_cfg.pool_port);
            if (!m_stratum.connect()) {
                util::log_warn("Connection failed, retrying in 5 s ...");
                std::this_thread::sleep_for(std::chrono::seconds(5));
                continue;
            }
        }
        m_stratum.run();   /* blocks until disconnect */
        if (m_running) {
            util::log_warn("Disconnected, reconnecting in 5 s ...");
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }
}

void Miner::mining_thread() {
    uint64_t nonce = 0;  /* will be randomised per-job in production */

    while (m_running) {
        /* wait for a job */
        stratum::Job job;
        kawpow::LightCache cache;
        {
            std::lock_guard<std::mutex> lock(m_job_mutex);
            if (!m_has_job) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            job   = m_current_job;
            cache = m_light_cache;
        }

        /* decode header & target */
        auto hdr_bytes = util::from_hex(job.header_hash);
        if (hdr_bytes.size() != 32) {
            util::log_warn("Invalid header hash length");
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        kawpow::hash256 header_hash;
        std::copy(hdr_bytes.begin(), hdr_bytes.end(), header_hash.begin());

        auto tgt_bytes = util::from_hex(job.target);
        kawpow::hash256 target{};
        if (tgt_bytes.size() == 32)
            std::copy(tgt_bytes.begin(), tgt_bytes.end(), target.begin());

#ifdef HAVE_CUDA
        /* GPU mining path */
        util::log_info("Mining on GPU %d  block=%llu", m_cfg.gpu_id,
                       (unsigned long long)job.block_number);

        cuda_mine(
            m_cfg.gpu_id,
            header_hash, nonce, job.block_number, target, cache,
            m_running, m_hash_count,
            [&](uint64_t sol_nonce, const kawpow::KawpowResult& result) {
                ++m_shares;
                m_stratum.submit(
                    job.job_id, sol_nonce,
                    job.header_hash,
                    util::to_hex(result.mix_hash.data(), 32));
            });

        nonce += m_hash_count.load();
#else
        /* CPU reference mining (slow – for verification only) */
        util::log_info("CPU mining  block=%llu (GPU build required for real speed)",
                       (unsigned long long)job.block_number);

        for (uint32_t i = 0; i < 256 && m_running; ++i, ++nonce) {
            auto result = kawpow::kawpow_hash(header_hash, nonce,
                                              job.block_number, cache);
            ++m_hash_count;

            if (kawpow::meets_target(result.final_hash, target)) {
                util::log_info("*** SHARE FOUND ***  nonce=%016llx",
                               (unsigned long long)nonce);
                ++m_shares;
                m_stratum.submit(
                    job.job_id, nonce,
                    job.header_hash,
                    util::to_hex(result.mix_hash.data(), 32));
            }
        }

        /* print hashrate periodically */
        if (m_hash_count % 1024 < 256)
            util::log_info("Hashrate: %.2f H/s  shares: %llu",
                           hashrate(), (unsigned long long)m_shares.load());
#endif
    }
}
