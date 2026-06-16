# Contributing to PRISM

Thanks for considering a contribution. PRISM is a local-first system that
holds extraordinarily personal data; we triage changes through that lens.

## Quick start

```bash
git clone https://github.com/chizoalban2003-beep/Prism.git
cd Prism
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                      # 2 800+ tests, runs in ~3 min
python3 -m ruff check .
python3 -m mypy --follow-imports=skip prism_daemon.py prism_asgi.py
```

## Ground rules

1. **One commit, one concern.** A single PR may contain several commits;
   each commit must make sense on its own.
2. **Tests are not optional** for new logic. Bug fixes need a regression
   test that fails without the fix.
3. **No silent fallbacks.** If something can't be done, raise or return
   a typed error — don't swallow exceptions and return empty.
4. **Security-sensitive paths.** Anything that touches the AST safety
   visitor, the SSRF guard, the chain prompt sanitiser, the federation
   HMAC path, or organ synthesis is reviewed at a higher bar. Walk
   adversarial cases in the PR description.
5. **Don't add cloud calls** without a reason. The whole point is local.

## Submitting changes

- Open a draft PR early if the change is non-trivial. Conversation in
  the PR is cheaper than a redesign after the fact.
- Run `pytest -q` and `ruff check .` locally before pushing.
- Reference the issue / Linear ticket if one exists.
- Keep the PR title under 70 characters.

## Reporting bugs

For non-security bugs, open a GitHub issue with:
- What you ran and what you expected.
- What actually happened (paste the traceback if any).
- Your platform, Python version, and `pip freeze` output.

For **security** issues, see `SECURITY.md` — do not open a public
issue.

## License

By contributing you agree your work is released under the MIT licence
of the project (`LICENSE`).
