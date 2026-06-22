# Code Sandbox — Pre-Deployment Security Audit (2026-06-21)

Adversarial red-team audit (7 attacker lenses → adversarial verification →
synthesis) run **before** deploying the sandbox live in the FedRAMP boundary.
57 serious findings surfaced; 8 survived verification as must-fix/exploitable.

## Verdict: conditional GO

The design is sound — the **real boundary is infrastructure** (zero-role SA +
egress denial + internal ingress), and the in-process seatbelt is correctly
labelled defense-in-depth. Confirmed: the seatbelt is bypassable three ways
(`_socket`, `importlib.reload(socket)`, `ctypes.CDLL`). So go-live is gated on
the code fixes below **and** verifying the load-bearing infra from inside the
deployed container.

## Code fixes — DONE (feature/sandbox-hardening)

- **CODE-1 (HIGH) — parent-memory balloon.** `communicate()` buffered all child
  stdout in the *parent* before the cap applied → a streaming child could OOM the
  parent. Fixed: bounded, deadline-driven selector read loop; accumulate ≤
  `2*OUTPUT_CAP`, then SIGKILL + `outcome="blocked"`. Test:
  `test_streaming_output_is_bounded_and_blocked`.
- **CODE-2 (HIGH) — post-kill hang.** The post-kill drain had no timeout; a forked
  grandchild (`os.fork`+`os.setsid`) that escaped the process group and held the
  pipe open wedged the `concurrency=1` worker forever. Fixed: the read loop is
  bounded by the wall-clock deadline + a short post-exit grace (never by EOF), and
  the child is reaped with a timeout. Test: `test_forked_grandchild_does_not_hang`.
- **CODE-3 (MED) — pre-screen + summary hygiene.** Added a static pre-screen
  (`tools/code_sandbox.py:_prescreen`) rejecting obvious escape primitives
  (`ctypes`, `_socket`, `os.system`, `subprocess`, `os.fork`, `os.posix_spawn`,
  `os.exec*`, `importlib.reload`) before the executor — closes the gap the LLM
  judge (intent-only) misses. And `result_summary` no longer persists raw
  stdout/stderr: it stores the structured `result` or a size summary, lightly
  redacted. Tests: `test_prescreen_*`, `test_result_summary_*`.

## Load-bearing infra — VERIFY at deploy (Task #2, gated)

Each is a single toggle that, if wrong, turns the bypassable seatbelt into a live
exfil/escalation path. **Verify from inside the deployed container**, not by
assuming defaults (Cloud Run does not deny egress by default).

- **INFRA-1 — zero-role SA real + attached** (most load-bearing). SA bindings must
  be EMPTY and the running service must use it (not the default compute SA, which
  has Editor). Gate the deploy on both; also strip `roles/editor` from the default
  compute SA. `gcloud projects get-iam-policy … --filter members:samurai-sandbox@…`
  → empty; `gcloud run services describe … --format='value(spec.template.spec.serviceAccountName)'`.
- **INFRA-2 — egress actually denied (end-to-end).** From a sandbox script, an
  outbound `socket.connect((public_ip,443))` must TIME OUT, not connect.
- **INFRA-3 — metadata server blocked.** `169.254.169.254` is link-local and
  reachable **regardless of egress routing** — the plan never specified blocking
  it. Untrusted code can always mint the SA token + read recon. Drop
  `169.254.0.0/16` at the container (iptables/nftables or a netns) or run a
  metadata-concealment posture. (With a verified zero-role SA the token authorizes
  nothing, but INFRA-1 + INFRA-3 together close it.)
- **INFRA-4 — Private Google Access OFF on the sandbox subnet.** "No NAT/internet
  route" is not enough: PGA / `restricted.googleapis.com` reaches `*.googleapis.com`
  with no internet route. Use a dedicated sandbox-only subnet with
  `privateIpGoogleAccess=False`; confirm the SA's project isn't in a VPC-SC
  perimeter granting in-boundary bucket read.
- **INFRA-5 — invoker IAM locked to the bot SA only.** `roles/run.invoker` members
  must be exactly `serviceAccount:samurai-bot@…` (internal ingress alone does not
  authenticate caller identity); no `allUsers`/`allAuthenticatedUsers`.
- **INFRA-6 — `SANDBOX_TOKEN` secret region-pinned** (user-managed, us-central1;
  org policy `constraints/gcp.resourceLocations`).

## Accepted residual risks (post-fix)

- In-process seatbelt fully bypassable — accepted **by design**; containment rests
  on INFRA-1/2/3/4. CODE-3 is the cheap backstop.
- Metadata recon disclosure if INFRA-3 is skipped — accepted only if INFRA-1 holds.
- DoS is self-contained to the optional, kill-switched `samurai-sandbox` service;
  the 2 Gi bot is not in the blast radius once CODE-1/CODE-2 land.
- The judge is not a code-safety scanner (intent-only, LLM-evadable) — accepted;
  CODE-3's static pre-screen + infra are the real controls.
