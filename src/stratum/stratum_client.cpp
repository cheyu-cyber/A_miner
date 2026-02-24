/*
 * stratum_client.cpp – Stratum protocol client for Ravencoin pool mining.
 *
 * Uses plain BSD sockets for portability.  JSON is handled with a
 * minimal inline parser (no external dependency).
 */
#include "stratum_client.h"
#include "utils/log.h"

#include <cstring>
#include <cstdio>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <poll.h>
#include <sstream>
#include <algorithm>

namespace stratum {

/* ---------- tiny JSON helpers (no dependency) ---------- */

static std::string json_string(const std::string& key, const std::string& json) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return {};
    pos = json.find(':', pos);
    if (pos == std::string::npos) return {};
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return {};
    auto end = json.find('"', pos + 1);
    if (end == std::string::npos) return {};
    return json.substr(pos + 1, end - pos - 1);
}

static int64_t json_int(const std::string& key, const std::string& json, int64_t def = 0) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return def;
    ++pos;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    char* endp = nullptr;
    int64_t val = std::strtoll(json.c_str() + pos, &endp, 10);
    return (endp && endp != json.c_str() + pos) ? val : def;
}

static bool json_bool(const std::string& key, const std::string& json, bool def = false) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return def;
    auto rest = json.substr(pos + 1);
    if (rest.find("true") < rest.find("false")) return true;
    if (rest.find("false") != std::string::npos) return false;
    return def;
}

/* ---------- StratumClient implementation ---------- */

StratumClient::StratumClient() = default;

StratumClient::~StratumClient() {
    disconnect();
}

void StratumClient::set_pool(const std::string& host, uint16_t port) {
    m_host = host;
    m_port = port;
}

void StratumClient::set_credentials(const std::string& user, const std::string& pass) {
    m_user = user;
    m_pass = pass;
}

void StratumClient::set_on_job(OnJobCallback cb) {
    m_on_job = std::move(cb);
}

bool StratumClient::connect() {
    if (m_host.empty() || m_port == 0) {
        util::log_error("Pool address not configured");
        return false;
    }

    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    std::string port_str = std::to_string(m_port);
    int err = getaddrinfo(m_host.c_str(), port_str.c_str(), &hints, &res);
    if (err != 0 || !res) {
        util::log_error("DNS resolution failed for %s: %s", m_host.c_str(), gai_strerror(err));
        return false;
    }

    m_sock = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (m_sock < 0) {
        util::log_error("socket() failed");
        freeaddrinfo(res);
        return false;
    }

    if (::connect(m_sock, res->ai_addr, res->ai_addrlen) < 0) {
        util::log_error("connect() to %s:%u failed", m_host.c_str(), m_port);
        close(m_sock);
        m_sock = -1;
        freeaddrinfo(res);
        return false;
    }
    freeaddrinfo(res);

    util::log_info("Connected to %s:%u", m_host.c_str(), m_port);
    m_connected = true;

    /* mining.subscribe */
    int id_sub = m_req_id++;
    char buf[512];
    snprintf(buf, sizeof(buf),
        "{\"id\":%d,\"method\":\"mining.subscribe\","
        "\"params\":[\"a_miner/1.0\",null,\"%s\",\"%s\"]}\n",
        id_sub, m_host.c_str(), port_str.c_str());
    if (!send_line(buf)) return false;

    /* mining.authorize */
    int id_auth = m_req_id++;
    snprintf(buf, sizeof(buf),
        "{\"id\":%d,\"method\":\"mining.authorize\","
        "\"params\":[\"%s\",\"%s\"]}\n",
        id_auth, m_user.c_str(), m_pass.c_str());
    return send_line(buf);
}

void StratumClient::disconnect() {
    m_connected = false;
    if (m_sock >= 0) {
        close(m_sock);
        m_sock = -1;
    }
}

bool StratumClient::is_connected() const {
    return m_connected;
}

bool StratumClient::submit(
    const std::string& job_id,
    uint64_t nonce,
    const std::string& header_hash,
    const std::string& mix_hash)
{
    if (!m_connected) return false;

    int id = m_req_id++;
    char buf[512];
    snprintf(buf, sizeof(buf),
        "{\"id\":%d,\"method\":\"mining.submit\","
        "\"params\":[\"%s\",\"%s\",\"%016llx\",\"%s\",\"%s\"]}\n",
        id, m_user.c_str(), job_id.c_str(),
        (unsigned long long)nonce, header_hash.c_str(), mix_hash.c_str());

    util::log_info("Submitting share: job=%s nonce=%016llx",
                   job_id.c_str(), (unsigned long long)nonce);
    return send_line(buf);
}

bool StratumClient::send_line(const std::string& json) {
    std::lock_guard<std::mutex> lock(m_send_mutex);
    ssize_t total = 0;
    ssize_t len = static_cast<ssize_t>(json.size());
    while (total < len) {
        ssize_t n = ::send(m_sock, json.c_str() + total, len - total, MSG_NOSIGNAL);
        if (n <= 0) {
            util::log_error("send() failed");
            disconnect();
            return false;
        }
        total += n;
    }
    return true;
}

void StratumClient::run() {
    std::string recv_buf;
    char tmp[4096];

    while (m_connected) {
        struct pollfd pfd{};
        pfd.fd     = m_sock;
        pfd.events = POLLIN;

        int ret = poll(&pfd, 1, 1000 /* ms */);
        if (ret < 0) {
            util::log_error("poll() error");
            disconnect();
            return;
        }
        if (ret == 0) continue;  /* timeout */

        ssize_t n = recv(m_sock, tmp, sizeof(tmp) - 1, 0);
        if (n <= 0) {
            util::log_error("Connection closed by pool");
            disconnect();
            return;
        }
        tmp[n] = '\0';
        recv_buf += tmp;

        /* process complete lines */
        size_t pos;
        while ((pos = recv_buf.find('\n')) != std::string::npos) {
            std::string line = recv_buf.substr(0, pos);
            recv_buf.erase(0, pos + 1);
            if (!line.empty())
                handle_line(line);
        }
    }
}

void StratumClient::handle_line(const std::string& line) {
    if (line.find("mining.notify") != std::string::npos) {
        handle_notify(line);
    } else {
        handle_response(line);
    }
}

void StratumClient::handle_notify(const std::string& line) {
    /*
     * Ravencoin kawpow stratum mining.notify params:
     *   [job_id, header_hash, seed_hash, target, clean_jobs, block_height, nbits]
     *
     * We extract them via simple positional parsing of the "params" array.
     */
    auto params_pos = line.find("\"params\"");
    if (params_pos == std::string::npos) return;

    auto arr_start = line.find('[', params_pos);
    if (arr_start == std::string::npos) return;

    /* collect quoted strings inside the params array */
    std::vector<std::string> values;
    size_t p = arr_start + 1;
    while (p < line.size() && line[p] != ']') {
        if (line[p] == '"') {
            auto end = line.find('"', p + 1);
            if (end == std::string::npos) break;
            values.push_back(line.substr(p + 1, end - p - 1));
            p = end + 1;
        } else if (line[p] == 't' || line[p] == 'f') {
            /* boolean */
            values.push_back(line.substr(p, 4 + (line[p] == 'f')));
            p += values.back().size();
        } else if (std::isdigit(line[p])) {
            auto end = line.find_first_of(",]", p);
            values.push_back(line.substr(p, end - p));
            p = end;
        } else {
            ++p;
        }
    }

    if (values.size() < 5) {
        util::log_warn("Malformed mining.notify (got %zu params)", values.size());
        return;
    }

    Job job;
    job.job_id       = values[0];
    job.header_hash  = values[1];
    job.seed_hash    = values[2];
    job.target       = values[3];
    job.clean_jobs   = (values[4] == "true");
    job.block_number = (values.size() > 5) ? std::stoull(values[5]) : 0;

    util::log_info("New job: id=%s height=%llu clean=%d",
                   job.job_id.c_str(),
                   (unsigned long long)job.block_number,
                   (int)job.clean_jobs);

    if (m_on_job)
        m_on_job(job);
}

void StratumClient::handle_response(const std::string& line) {
    /* Check for error responses */
    if (line.find("\"error\"") != std::string::npos &&
        line.find("null") == std::string::npos)
    {
        auto err_msg = json_string("error", line);
        if (!err_msg.empty())
            util::log_warn("Pool error: %s", err_msg.c_str());
    }

    /* accepted / rejected share feedback */
    if (line.find("\"result\"") != std::string::npos) {
        if (line.find("true") != std::string::npos)
            util::log_info("Share accepted");
        else if (line.find("false") != std::string::npos)
            util::log_warn("Share rejected");
    }
}

} // namespace stratum
