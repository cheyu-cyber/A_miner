/*
 * main.cpp – Entry point for the A_miner Ravencoin (KawPow) miner.
 *
 * Usage:
 *   a_miner -o <pool:port> -u <wallet.worker> [-p <password>] [-d <gpu_id>]
 */
#include "miner.h"
#include "utils/log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <atomic>

static std::atomic<bool> g_running{true};

static void signal_handler(int) {
    g_running = false;
}

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "A_miner – Ravencoin (KawPow) GPU miner\n"
        "\n"
        "Usage:\n"
        "  %s -o <pool:port> -u <wallet.worker> [options]\n"
        "\n"
        "Options:\n"
        "  -o <host:port>   Pool address (e.g. stratum+tcp://rvn.2miners.com:6060)\n"
        "  -u <wallet>      Wallet address or user.worker\n"
        "  -p <password>    Worker password (default: x)\n"
        "  -d <id>          CUDA device ID (default: 0)\n"
        "  -v               Verbose logging (debug level)\n"
        "  -h               Show this help\n"
        "\n"
        "Example:\n"
        "  %s -o rvn.2miners.com:6060 -u RVNaddress.rig1\n",
        prog, prog);
}

int main(int argc, char* argv[]) {
    Miner::Config cfg;
    cfg.worker_pass = "x";
    cfg.gpu_id = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            std::string addr = argv[++i];
            /* strip protocol prefix if present */
            auto pos = addr.find("://");
            if (pos != std::string::npos)
                addr = addr.substr(pos + 3);
            auto colon = addr.rfind(':');
            if (colon != std::string::npos) {
                cfg.pool_host = addr.substr(0, colon);
                cfg.pool_port = static_cast<uint16_t>(std::atoi(addr.substr(colon + 1).c_str()));
            } else {
                cfg.pool_host = addr;
                cfg.pool_port = 6060;  /* common Ravencoin stratum port */
            }
        } else if (std::strcmp(argv[i], "-u") == 0 && i + 1 < argc) {
            cfg.wallet = argv[++i];
        } else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            cfg.worker_pass = argv[++i];
        } else if (std::strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            cfg.gpu_id = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            util::set_log_level(util::LogLevel::DEBUG);
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (cfg.pool_host.empty() || cfg.wallet.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    /* print banner */
    std::printf(
        "\n"
        "  ╔═══════════════════════════════════════════╗\n"
        "  ║   A_miner – Ravencoin KawPow GPU Miner   ║\n"
        "  ╚═══════════════════════════════════════════╝\n"
        "\n"
        "  Pool:   %s:%u\n"
        "  Wallet: %s\n"
        "  GPU:    %d\n"
        "\n",
        cfg.pool_host.c_str(), cfg.pool_port,
        cfg.wallet.c_str(), cfg.gpu_id);

    /* trap signals for clean shutdown */
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    Miner miner(cfg);
    miner.start();

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        util::log_info("Hashrate: %.2f H/s | Shares: %llu",
                       miner.hashrate(),
                       (unsigned long long)miner.shares_found());
    }

    miner.stop();
    util::log_info("Bye.");
    return 0;
}
