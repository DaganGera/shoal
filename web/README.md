# SHOAL landing page

A single static page (`index.html`) for showing SHOAL to a jury: what it does, the
pipeline, the outputs, and the rigor argument. Separate from the Streamlit
dashboard (`app/dashboard.py`), which is the working tool.

## Run it

```bash
python3 -m http.server 4173 --directory web
# open http://localhost:4173
```

No runtime dependencies. Deploy by copying `web/` to any static host; a GitHub
Pages workflow (`.github/workflows/pages.yml`) does this on every push to `web/**`.

## Styling

Built on **Tailwind CSS v4 + DaisyUI v5** under one locked custom theme (`shoal`):
a cool near-black base, a single amber accent taken from the tracker overlay,
Geist + Geist Mono. The design brief and section-by-section rationale are in the
comment at the top of `index.html`.

`web/styles.css` is the built stylesheet and **is committed**, so the page needs
no build step to serve or deploy. Regenerate it after editing `index.html` or
`src/input.css`:

```bash
web/build.sh
```

`build.sh` fetches a pinned standalone Tailwind CLI plus the DaisyUI package
tarball into `web/.cache/` (~110 MB, gitignored) with `curl`. No npm or Node
required. Linux and macOS, x64 and arm64.

## Assets

Everything in `web/assets/` is generated from real run directories, not invented:

```bash
uv run python web/build_assets.py     # needs outputs/{reef_clear,reef_turbid_matched,anemone_static,reef_snapper}/
```

produces: `tracking.mp4` + `tracking_poster.jpg` (annotated clip loop), `residency.png`,
`trajectories.png`, `visibility.png`, `ab_raw.jpg` / `ab_restored.jpg`, and
`data.json` (the numbers bundle). Self-hosted Geist / Geist Mono woff2 files live
in `assets/fonts/`.
