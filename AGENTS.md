# AGENTS.md

## Project

This is a Zola static personal site. Markdown content lives in `content/`,
structured content and generated manifests in `data/`, Tera views in
`templates/`, styles in `sass/`, served assets in `static/`, and maintenance
utilities in `scripts/`. `public/` is generated output; do not edit it.

Main features are the academic profile and publications, technical blog,
Outside galleries, citation data, site search, Atom feeds, responsive styling,
and light/dark/system themes.

## Constraints

- Use `./serve.sh` on Unix or `.\serve.ps1` on Windows for development. Never
  invoke `zola serve` directly; the wrappers also watch `data/`.
- Keep camera originals and other large source assets out of Git. Follow
  `docs/assets-and-galleries.md` and commit only web-ready outputs.
- Preserve existing content/front-matter conventions and keep templates valid
  Tera. Run `zola build` when a production-build check is appropriate.
- Do not hand-edit generated files in `public/`.
- Preserve copyright, license, and third-party attribution notices. See
  `LICENSE.md` before adding or reusing content or assets.

## Agent debugging server lifecycle

Before browser or live-reload debugging, check for an existing Zola server
(process and listening port, normally 1111).

- If one exists, reuse it. Do not start another server and do not stop or
  restart the existing process.
- If none exists, start the appropriate `serve.sh`/`serve.ps1` wrapper, record
  the process you started, and use it for debugging. Stop only that process—and
  confirm it has stopped—before replying.
- Exception: if the user explicitly asks for a live preview, start or reuse a
  server, leave it available, and report its URL. Never stop a server that was
  already running before the task.
