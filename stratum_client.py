"""
stratum_client.py – Stratum V1 client for Ravencoin (KawPow) mining pools.

Handles:
  - TCP connection and reconnection
  - JSON-RPC framing (newline-delimited)
  - mining.subscribe / mining.authorize handshake
  - mining.notify (incoming work)
  - mining.set_difficulty
  - mining.submit (share submission)
"""

import json
import logging
import socket
import struct
import threading
import time
from typing import Optional, Tuple

logger = logging.getLogger("A_miner.stratum")

# Seconds between reconnect attempts
RECONNECT_DELAY  = 5
# Socket read timeout (seconds)
SOCKET_TIMEOUT   = 30
# How many nonce batches to try before checking for new work
BATCHES_PER_WORK = 256


class StratumClient:
    """
    Stratum V1 mining client for KawPow / Ravencoin pools.
    """

    def __init__(
        self,
        host:    str,
        port:    int,
        wallet:  str,
        password: str,
        worker:  str,
        kawpow,
    ):
        self.host     = host
        self.port     = port
        self.wallet   = wallet
        self.password = password
        self.worker   = worker
        self.kawpow   = kawpow

        self._sock:       Optional[socket.socket] = None
        self._recv_buf:   str   = ""
        self._req_id:     int   = 0
        self._running:    bool  = False
        self._lock        = threading.Lock()

        # Current work
        self._job_id:     Optional[str]   = None
        self._header:     Optional[bytes] = None   # 32-byte header hash
        self._seed:       Optional[bytes] = None   # 32-byte seed hash
        self._target:     Optional[bytes] = None   # 32-byte target
        self._block_num:  int  = 0
        self._difficulty: float = 1.0
        self._work_event  = threading.Event()

        # Stats
        self._shares_accepted = 0
        self._shares_rejected = 0
        self._hashes_done     = 0
        self._start_time      = 0.0

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def start(self) -> None:
        """Start the receive thread; connect to the pool."""
        self._running     = True
        self._start_time  = time.time()
        t = threading.Thread(target=self._run, daemon=True, name="stratum-recv")
        t.start()
        self._recv_thread = t

    def stop(self) -> None:
        """Signal all threads to stop."""
        self._running = False
        self._work_event.set()
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass

    def wait(self) -> None:
        """Block until interrupted (runs the mining loop on the caller thread)."""
        try:
            self._mining_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    # -----------------------------------------------------------------------
    # Internal – connection management
    # -----------------------------------------------------------------------
    def _connect(self) -> bool:
        """Attempt to connect to pool. Returns True on success."""
        try:
            logger.info("Connecting to %s:%d…", self.host, self.port)
            sock = socket.create_connection((self.host, self.port),
                                            timeout=SOCKET_TIMEOUT)
            sock.settimeout(SOCKET_TIMEOUT)
            with self._lock:
                self._sock     = sock
                self._recv_buf = ""
            logger.info("Connected.")
            return True
        except OSError as exc:
            logger.error("Connection failed: %s", exc)
            return False

    def _send(self, msg: dict) -> None:
        """Serialise *msg* to JSON and send over the socket."""
        with self._lock:
            if self._sock is None:
                return
            data = json.dumps(msg) + "\n"
            try:
                self._sock.sendall(data.encode())
            except OSError as exc:
                logger.warning("Send error: %s", exc)

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    # -----------------------------------------------------------------------
    # Internal – handshake
    # -----------------------------------------------------------------------
    def _subscribe(self) -> None:
        self._send({
            "id":     self._next_id(),
            "method": "mining.subscribe",
            "params": ["A_miner/1.0"],
        })

    def _authorize(self) -> None:
        user = f"{self.wallet}.{self.worker}"
        self._send({
            "id":     self._next_id(),
            "method": "mining.authorize",
            "params": [user, self.password],
        })

    # -----------------------------------------------------------------------
    # Internal – receive loop (runs in background thread)
    # -----------------------------------------------------------------------
    def _run(self) -> None:
        while self._running:
            if not _attempt_connect(self):
                time.sleep(RECONNECT_DELAY)
                continue

            self._subscribe()
            self._authorize()

            while self._running:
                try:
                    chunk = self._sock.recv(4096)
                except socket.timeout:
                    continue
                except OSError as exc:
                    logger.warning("Socket error: %s – reconnecting…", exc)
                    break

                if not chunk:
                    logger.warning("Server closed connection – reconnecting…")
                    break

                with self._lock:
                    self._recv_buf += chunk.decode(errors="replace")

                self._process_buffer()

            with self._lock:
                if self._sock:
                    try:
                        self._sock.close()
                    except OSError:
                        pass
                    self._sock = None

            if self._running:
                logger.info("Reconnecting in %ds…", RECONNECT_DELAY)
                time.sleep(RECONNECT_DELAY)

    def _process_buffer(self) -> None:
        while True:
            with self._lock:
                pos = self._recv_buf.find("\n")
                if pos == -1:
                    break
                line           = self._recv_buf[:pos]
                self._recv_buf = self._recv_buf[pos + 1:]

            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Invalid JSON: %r", line)
                continue

            self._handle_message(msg)

    def _handle_message(self, msg: dict) -> None:
        method = msg.get("method", "")
        result = msg.get("result")
        err    = msg.get("error")

        if method == "mining.notify":
            self._on_notify(msg.get("params", []))
        elif method == "mining.set_difficulty":
            self._on_set_difficulty(msg.get("params", [1.0]))
        elif result is not None:
            # Response to subscribe/authorize/submit
            if err:
                logger.warning("Server error: %s", err)
            else:
                req_id = msg.get("id")
                logger.debug("Server response id=%s: %s", req_id, result)
                if isinstance(result, bool) and result:
                    logger.debug("Request %s accepted.", req_id)
        else:
            logger.debug("Unknown message: %s", msg)

    # -----------------------------------------------------------------------
    # Stratum message handlers
    # -----------------------------------------------------------------------
    def _on_set_difficulty(self, params: list) -> None:
        if params:
            self._difficulty = float(params[0])
            logger.info("Difficulty set to %g", self._difficulty)

    def _on_notify(self, params: list) -> None:
        """
        KawPow notify format:
          params[0] = job_id          (str)
          params[1] = header_hash     (64-char hex)
          params[2] = seed_hash       (64-char hex)
          params[3] = target          (64-char hex)
          params[4] = clean_jobs      (bool)
          params[5] = block_number    (int or hex str)
        """
        if len(params) < 5:
            logger.warning("Malformed notify: %s", params)
            return

        try:
            job_id      = params[0]
            header_hex  = params[1]
            seed_hex    = params[2]
            target_hex  = params[3]
            clean_jobs  = bool(params[4])
            block_num   = int(params[5], 16) if isinstance(params[5], str) else int(params[5])
        except (IndexError, ValueError) as exc:
            logger.warning("Notify parse error: %s  params=%s", exc, params)
            return

        self._job_id    = job_id
        self._header    = bytes.fromhex(header_hex)
        self._seed      = bytes.fromhex(seed_hex)
        self._target    = bytes.fromhex(target_hex)
        self._block_num = block_num

        logger.info("New work  job=%s  block=%d  clean=%s",
                    job_id, block_num, clean_jobs)

        self._work_event.set()

    def _difficulty_to_target(self) -> bytes:
        """
        Convert pool difficulty to a 256-bit target (big-endian).
        target = 2^256 / difficulty / 2^32
        """
        if self._target:
            return self._target
        max_target = (1 << 256) - 1
        target_int = max_target // max(1, int(self._difficulty * (1 << 32)))
        return target_int.to_bytes(32, "big")

    # -----------------------------------------------------------------------
    # Share submission
    # -----------------------------------------------------------------------
    def _submit_share(
        self,
        job_id:   str,
        nonce:    int,
        header:   bytes,
        mix_hash: bytes,
    ) -> None:
        nonce_hex    = f"0x{nonce:016x}"
        header_hex   = "0x" + header.hex()
        mix_hash_hex = "0x" + mix_hash.hex()
        user         = f"{self.wallet}.{self.worker}"

        self._send({
            "id":     self._next_id(),
            "method": "mining.submit",
            "params": [user, job_id, nonce_hex, header_hex, mix_hash_hex],
        })
        logger.info("Share submitted  nonce=%s  job=%s", nonce_hex, job_id)

    # -----------------------------------------------------------------------
    # Mining loop (runs on the main thread)
    # -----------------------------------------------------------------------
    def _mining_loop(self) -> None:
        logger.info("Mining loop started. Waiting for work…")
        base_nonce    = 0
        last_stats_t  = time.time()

        while self._running:
            # Wait for valid work
            if self._job_id is None:
                self._work_event.wait(timeout=5)
                self._work_event.clear()
                continue

            # Snapshot current work
            job_id     = self._job_id
            header     = self._header
            block_num  = self._block_num
            target     = self._difficulty_to_target()

            self._work_event.clear()

            # Run a batch of nonces on the GPU
            try:
                winning_nonce, mix_hash = self.kawpow.mine(
                    header_hash  = header,
                    block_number = block_num,
                    target       = target,
                    base_nonce   = base_nonce,
                )
            except Exception as exc:
                logger.error("Mining error: %s", exc, exc_info=True)
                time.sleep(1)
                continue

            self._hashes_done += self.kawpow.NONCES_PER_BATCH
            base_nonce        += self.kawpow.NONCES_PER_BATCH

            if winning_nonce is not None:
                logger.info("Share found!  nonce=0x%016x", winning_nonce)
                self._submit_share(job_id, winning_nonce, header, mix_hash)
                self._shares_accepted += 1

            # Check if new work arrived
            if self._work_event.is_set() or self._job_id != job_id:
                logger.debug("New work received, switching…")
                base_nonce = 0
                continue

            # Periodic stats
            now = time.time()
            if now - last_stats_t >= 30:
                elapsed  = now - self._start_time
                hashrate = self._hashes_done / elapsed / 1e6
                logger.info(
                    "Hashrate %.2f MH/s  accepted=%d  rejected=%d",
                    hashrate, self._shares_accepted, self._shares_rejected,
                )
                last_stats_t = now


# ---------------------------------------------------------------------------
# Helper used inside _run() to avoid a forward-reference
# ---------------------------------------------------------------------------
def _attempt_connect(client: "StratumClient") -> bool:
    return client._connect()
