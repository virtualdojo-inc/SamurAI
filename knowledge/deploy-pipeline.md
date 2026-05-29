---
title: Deploy Pipeline
summary: How code changes reach the live SamurAI bot — push to main triggers a tested, blue/green Cloud Run deploy with health-gating and auto-rollback.
tags: [ci-cd, deploy, cloud-run]
updated: 2026-05-29
---

# Deploy Pipeline

Pushing to `main` (repo `virtualdojo-inc/SamurAI`) triggers
`.github/workflows/deploy.yml`:

1. **Test gate** — full `pytest` suite must pass.
2. **Blue/green deploy** — deploy the new revision with `--no-traffic --tag`,
   health-check the candidate's tagged `/health` URL while production stays on
   the current revision, promote to 100% only if healthy, re-verify, and
   **auto-rollback** to the previous revision if the post-promote check fails.

Auth is keyless via Workload Identity Federation (no SA key in GitHub). Because a
failed candidate never receives traffic, a bad build cannot take the bot down.

The self-improvement loop ([[self-improvement-loop]] once written) opens PRs that,
on merge, ride this same pipeline. Infra context: [[virtualdojo-infra]].

## Related
- [[virtualdojo-infra]]
