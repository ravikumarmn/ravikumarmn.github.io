# ravikumarmn.github.io

Personal site of Ravikumar Naduvin, Senior AI Engineer. Static Astro site deployed to GitHub Pages.

## Develop

```sh
nvm use            # Node 22+
npm install
npm run dev        # http://localhost:4321 (inferred copy is underlined in dev)
npm run build      # type-check + build to dist/
npm run preview    # serve dist/
npm run audit:content   # list remaining TODO placeholders and copy to verify
```

## Edit content (no component changes needed)

| What | File |
| --- | --- |
| Name, headline, links, email, education | `src/content/site/profile.json` |
| Impact numbers (each has a `source`) | `src/content/site/metrics.json` |
| Experience timeline | `src/content/site/experience.json` |
| Skills | `src/content/site/skills.json` |
| "Also built" projects | `src/content/site/projects.json` |
| Projects (homepage + `/projects`) | `src/content/site/building.json` |
| About paragraphs | `src/content/site/about.json` |
| Case studies | `src/content/work/*.mdx` |
| Architecture diagrams | `src/components/diagrams/specs.ts` |
| Blog posts | `src/content/posts/*.md` (served at `/blogs/<slug>`) |

JSON is validated at build time (`src/lib/site.ts`, `src/content.config.ts`), so a typo fails the build with a clear message.

**Placeholders.** Any JSON value starting with `TODO` and any `<Todo>` in MDX renders as a visible amber badge, and the deploy workflow builds with `STRICT_CONTENT=1`, so a page with one fails to deploy. Facts that would strengthen a case study but aren't on the résumé are kept as hidden `{/* GAP: … */}` comments. `<Verify>` marks copy that was inferred from the résumé; it is underlined in dev only. `npm run audit:content` lists all three.

**Résumé.** The Résumé buttons open the Google Drive link in `profile.json` (`resume`); `/resume` redirects there too, so it works as a short link. Update the file in Drive with **Manage versions → Upload new version** to keep the same URL. The local copy in `resume/` is gitignored.

## Deploy

Pushing to `main` runs `.github/workflows/deploy.yml` (build, then deploy to GitHub Pages).
One-time setup: **Settings → Pages → Build and deployment → Source: GitHub Actions**.

### Custom domain

1. Add `public/CNAME` containing the domain (e.g. `ravikumar.dev`).
2. Set `site` in `astro.config.mjs` to `https://ravikumar.dev`, and update the URL in `public/robots.txt`.
3. DNS: apex → `A` records `185.199.108.153`, `185.199.109.153`, `185.199.110.153`, `185.199.111.153`; `www` → `CNAME ravikumarmn.github.io`.
4. In Settings → Pages, set the custom domain and enable **Enforce HTTPS**.

## Notes

- Old Jekyll post URLs (`/blogs/<slug>`) are preserved exactly via `build.format: 'file'`.
- `public/sw.js` unregisters the service worker the old Jekyll theme could install.
- OG images are generated at build time (`src/pages/og/[...path].png.ts`).
- No analytics or tracking.
