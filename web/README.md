# SHOAL demo page

A single static page (`index.html`) for showing SHOAL to a jury: what it does, the
pipeline, the outputs, and the rigor argument. Separate from the Streamlit
dashboard (`app/dashboard.py`), which is the working tool.

## Run it

```bash
python3 -m http.server 4173 --directory web
# open http://localhost:4173
```

No build step, no dependencies. Deploy by copying `web/` to any static host
(GitHub Pages, Netlify, Vercel, an S3 bucket).

## Assets

Everything in `web/assets/` is generated from real run directories, not invented:

```bash
uv run python web/build_assets.py     # needs outputs/{reef_clear,reef_turbid_matched,anemone_static,reef_snapper}/
```

produces: `tracking.mp4` (annotated clip loop), `residency.png`, `trajectories.png`,
`visibility.png`, `ablation.png`, `ab_raw.jpg` / `ab_restored.jpg`, and `data.json`
(the numbers bundle). Self-hosted Geist / Geist Mono woff2 files live in
`assets/fonts/`.

## Design notes

Committed dark theme, single amber accent taken from the tracker overlay, Geist +
Geist Mono, IntersectionObserver scroll reveals that collapse to static under
`prefers-reduced-motion`. Built to the `design-taste-frontend` skill: no build
scaffold since npm is unavailable here and a static page is the right shape for a
projector demo anyway.
