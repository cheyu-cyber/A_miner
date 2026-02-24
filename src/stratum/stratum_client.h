/*
 * stratum_client.h – Stratum protocol client for Ravencoin (KawPow) mining.
 *
 * Implements the subset of stratum needed for pool mining:
 *   mining.subscribe  – register with the pool
 *   mining.authorize  – authenticate with worker credentials
 *   mining.notify     – receive new work
 *   mining.submit     – submit solutions
 */
#pragma once

#include <cstdint>
#include <string>
#include <functional>
#include <mutex>
#include <atomic>
#include <thread>

namespace stratum {

/* Represents a mining job received from the pool */
struct Job {
    std::string  job_id;
    std::string  header_hash;   /* hex-encoded 32-byte header hash */
    std::string  seed_hash;     /* hex-encoded 32-byte seed hash   */
    std::string  target;        /* hex-encoded 32-byte target      */
    uint64_t     block_number;
    bool         clean_jobs;    /* discard stale work? */
};

/* Callback invoked when a new job arrives */
using OnJobCallback = std::function<void(const Job&)>;

class StratumClient {
public:
    StratumClient();
    ~StratumClient();

    /* Configuration */
    void set_pool(const std::string& host, uint16_t port);
    void set_credentials(const std::string& user, const std::string& pass);
    void set_on_job(OnJobCallback cb);

    /* Lifecycle */
    bool connect();
    void disconnect();
    bool is_connected() const;

    /* Submit a solution back to the pool */
    bool submit(const std::string& job_id,
                uint64_t nonce,
                const std::string& header_hash,
                const std::string& mix_hash);

    /* Blocking event loop – call from a dedicated thread */
    void run();

private:
    /* Internal helpers */
    bool send_line(const std::string& json);
    void handle_line(const std::string& line);
    void handle_notify(const std::string& line);
    void handle_response(const std::string& line);

    std::string  m_host;
    uint16_t     m_port = 0;
    std::string  m_user;
    std::string  m_pass;
    int          m_sock = -1;
    std::atomic<bool> m_connected{false};
    std::atomic<int>  m_req_id{1};
    OnJobCallback     m_on_job;
    mutable std::mutex m_send_mutex;
};

} // namespace stratum
