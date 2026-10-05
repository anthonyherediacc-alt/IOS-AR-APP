# AR Wound Sim (Safari prototype)

Camera-based pseudo-AR: MediaPipe Hand Landmarker tracks a hand in the browser and a PNG wound is drawn over it with Canvas 2D. Static site, no build step, no backend.

## Run
Must be served over HTTPS for camera access (or `http://localhost` on desktop).

- Local: `npx serve .` or `python3 -m http.server`, open `http://localhost:8000`.
- iPhone: deploy the repo root to GitHub Pages / Netlify / Cloudflare Pages and open the HTTPS URL in Safari.

## Add a wound
1. Put a transparent PNG/WebP in `assets/wounds/` (draw it with the fingers pointing "up").
2. Add one entry to `WOUNDS` in `js/config.js`.
