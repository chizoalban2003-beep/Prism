# Security Policy

PRISM runs **locally** and holds an enormous amount of personal context —
decision history, beliefs, identity inferences, federated peer secrets, and
any organ output that has touched user data. A vulnerability here is more
than a remote-code-execution issue: it is a leak of *who you are* to a
recipient who is allowed to act on your behalf.

We take that seriously. If you find a security issue, please tell us
**privately** before disclosing publicly.

## Supported versions

Until v1.0, only the latest tagged release on the `main` branch is
supported. Older tags receive no backports.

| Version | Supported |
| ------- | --------- |
| `0.x.y` (latest) | ✅ |
| anything older | ❌ |

## Reporting a vulnerability

Email **chizoalban2003@gmail.com** with the subject line
`PRISM SECURITY: <one-line summary>`. Include:

1. What you found and where (file:line is great).
2. A minimal reproduction — code, request, or steps.
3. The impact you believe it has (what an attacker gains).
4. Your suggested fix, if any.

We will acknowledge within **72 hours** and aim to ship a fix within
**14 days** for HIGH/CRITICAL issues. We will credit you in the
CHANGELOG and release notes unless you ask us not to.

Please do **not** open a public GitHub issue for a security report.

## What's in-scope

* Bypasses of the AST safety check in `prism_autonomous` or
  `prism_organ_loader`.
* SSRF gadgets in `prism_browser_agent`, `prism_federation`, or organs
  that fetch URLs.
* Prompt injection that escapes the `<<<USER_INPUT>>>` / `<<<EVIDENCE>>>`
  sanitisation in `prism_chain`.
* Federation auth / HMAC bypasses.
* Path-traversal or arbitrary-write in any organ.
* Token / secret exposure (`~/.prism/auth_token`, env vars, log files).

## What's out of scope

* DNS rebinding against the SSRF guard's resolve-then-fetch race — we
  document this as a known limitation; the mitigation requires a custom
  HTTP resolver, which is on the v0.2 roadmap.
* Denial of service against a single user's own local daemon.
* Findings that require an attacker who already holds the bearer token.
