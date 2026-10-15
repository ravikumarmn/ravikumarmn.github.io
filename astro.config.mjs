// @ts-check
import { defineConfig, fontProviders } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { unified } from '@astrojs/markdown-remark';
import tailwindcss from '@tailwindcss/vite';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

/** Overflowing display math scrolls horizontally, so it must be keyboard-focusable. */
function rehypeFocusableMath() {
  /** @param {any} node */
  const walk = (node) => {
    if (node.type === 'element' && node.properties?.className?.includes?.('katex-display')) {
      node.properties.tabIndex = 0;
    }
    node.children?.forEach(walk);
  };
  return walk;
}

/** @param {string} pkg @param {string} file */
const fontFile = (pkg, file) => `./node_modules/@fontsource-variable/${pkg}/files/${file}`;

export default defineConfig({
  // User-site repo (ravikumarmn.github.io), so the site is served from the root.
  // For a custom domain: set `site` to it and add public/CNAME (see README).
  site: 'https://ravikumarmn.github.io',
  base: '/',
  // `file` output + no trailing slash keeps the old Jekyll post URLs
  // (/blogs/<slug>) resolving exactly as before, without redirects.
  trailingSlash: 'never',
  build: { format: 'file' },
  markdown: {
    processor: unified({
      remarkPlugins: [[remarkMath, { singleDollarTextMath: false }]],
      rehypePlugins: [[rehypeKatex, { strict: false, throwOnError: false }], rehypeFocusableMath],
    }),
    shikiConfig: {
      themes: { light: 'github-light-default', dark: 'github-dark-dimmed' },
      defaultColor: false,
    },
  },
  fonts: [
    {
      name: 'Geist',
      cssVariable: '--font-geist',
      provider: fontProviders.local(),
      fallbacks: ['ui-sans-serif', 'system-ui', 'sans-serif'],
      options: {
        variants: [{ src: [fontFile('geist', 'geist-latin-wght-normal.woff2')], weight: '100 900', style: 'normal' }],
      },
    },
    {
      name: 'Geist Mono',
      cssVariable: '--font-geist-mono',
      provider: fontProviders.local(),
      fallbacks: ['ui-monospace', 'monospace'],
      options: {
        variants: [{ src: [fontFile('geist-mono', 'geist-mono-latin-wght-normal.woff2')], weight: '100 900', style: 'normal' }],
      },
    },
  ],
  redirects: {
    // Short link to the résumé on Google Drive (same URL as profile.json → resume).
    '/resume': 'https://drive.google.com/file/d/15pZ_vc9nsAT1Wo9ld3PBmUD5Tumd7yJP/view?usp=sharing',
    '/about': '/#about',
    '/archives': '/writing',
    '/tags': '/writing',
    '/categories': '/writing',
  },
  integrations: [
    mdx(),
    sitemap({ filter: (page) => !/\/(404|about|archives|tags|categories|resume)$/.test(page) }),
  ],
  vite: { plugins: [tailwindcss()] },
});
