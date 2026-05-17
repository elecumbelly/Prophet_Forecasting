# Security Policy

## Reporting a Vulnerability

If you discover a security issue in this project, please email the maintainer
(see the GitHub profile) rather than opening a public issue. Include:

- A description of the issue and where it lives in the code.
- A minimal reproduction (a curl invocation, a request payload, or a code
  snippet).
- Your assessment of the impact.

We aim to acknowledge reports within 72 hours.

## Supported Versions

The project is on `main` only — there are no maintained release branches.
Security fixes land on `main` and are immediately the recommended version.

## Threat Model

The Flask API is intended to run inside a trusted network or behind an
authenticated reverse proxy. It does not implement user authentication or
authorization itself. The following protections ARE in place:

- **CORS allowlist** via `CORS_ALLOWED_ORIGINS`; never `*`.
- **Strict identifier validation** (`[A-Za-z_][A-Za-z0-9_]{0,62}`) on every
  `table` / `ts_column` / `y_column` / regressor input across the CLI, the
  `/api/*` routes, the service layer, and the data loader.
- **Column existence check** on `/api/historical_data` and `/api/forecast`
  via `inspect()` before issuing any query.
- **Required `FLASK_SECRET_KEY`** in production (`FLASK_ENV=production`); the
  app refuses to start without it.
- **Path-traversal-resistant static route** `/outputs_web/<file>` that
  whitelists `.png`/`.csv`/`.json` extensions and rejects `..` components.
- **Model JSON cache** stored outside the publicly served directory
  (`models_cache/` is a sibling of `outputs_web/`).
- **Per-IP rate limit** on auto-tune (`auto_tune=true`) forecast runs.

## Dependency Hygiene

This project pins lower bounds on direct dependencies in `pyproject.toml` and
on frontend dependencies in `frontend/package.json`. Transitive vulnerability
reports surface through:

- **`pip-audit`** for Python deps.
- **`pnpm audit --prod`** for frontend deps.
- **`trivy fs`** scanning lockfiles for both ecosystems.
- **`bandit`** + **`semgrep`** + **`gitleaks`** for source-level issues.

### Known transitive issues (tracked, not blocking)

| Package           | Source                    | CVE class                                  | Status                                                  |
| ----------------- | ------------------------- | ------------------------------------------ | ------------------------------------------------------- |
| `pillow`          | transitive of `matplotlib`| Out-of-bounds write in PSD/FITS decoders   | Not in our execution path (we only `savefig`). Bumps automatically when `matplotlib` updates its floor. |
| `postcss`         | transitive of `next`      | XSS via unescaped `</style>` in stringify  | Clears when Next.js publishes a `postcss` floor bump.   |

We do not pin these directly because:
1. They are not direct dependencies (changes ripple through the upstream).
2. The vulnerable code paths are not used by this project.
3. Pinning would risk creating resolver conflicts with the upstream package.

If either becomes exploitable in our execution path, we will pin the floor.

## Local audit workflow

Before pushing, contributors run `/eod` (an internal audit skill) which runs
all five scanners and reports delta-introduced findings. CI runs a subset of
the same checks (`ruff`, `vulture`, `mypy`, `pytest`, `pnpm lint`,
`pnpm typecheck`, `pnpm build`) on every push and pull request.

## Out of Scope

- **Authentication / authorization**: the API expects a trusted client (e.g.,
  a Next.js SPA running behind your reverse proxy). If you expose `/api/*`
  to the public internet, put it behind your own auth gateway.
- **Long-lived secrets**: the only secret the app needs is
  `FLASK_SECRET_KEY` for session signing. PostgreSQL credentials come from
  `.env`. Neither is committed to the repo.
- **Denial of service from arbitrary auto-tune requests**: mitigated by the
  per-IP rate limit, but a determined attacker could still drive CPU usage
  on a forecast worker. Recommended deployment runs forecast workers as a
  separate process pool that the API submits jobs to.
