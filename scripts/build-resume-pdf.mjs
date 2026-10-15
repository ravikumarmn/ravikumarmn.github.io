#!/usr/bin/env node
/**
 * Prints the built /resume page to dist/resume.pdf, so the PDF always matches the site's content.
 * Runs after `astro build`. Needs a local Chrome or Chromium (preinstalled on GitHub's Ubuntu runners);
 * set CHROME_PATH to point at a different binary.
 *
 * Fails the build if the page can't be printed or its HTML contains anything that looks like a
 * phone number, so a broken or private résumé can't ship.
 */
import { existsSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';
import puppeteer from 'puppeteer-core';

const DIST = join(process.cwd(), 'dist');
const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css',
  '.js': 'text/javascript',
  '.woff2': 'font/woff2',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
};

const chrome = [
  process.env.CHROME_PATH,
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
].find((path) => path && existsSync(path));
if (!chrome) {
  console.error('build-resume-pdf: no Chrome found. Install Chrome or set CHROME_PATH.');
  process.exit(1);
}

// Serve dist/ the way GitHub Pages does (build.format 'file': /resume → resume.html).
const server = createServer((req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^(\.\.[/\\])+/, '');
  const candidates = [join(DIST, path), join(DIST, `${path}.html`), join(DIST, path, 'index.html')];
  const file = candidates.find((candidate) => existsSync(candidate) && statSync(candidate).isFile());
  if (!file) {
    res.writeHead(404).end();
    return;
  }
  res.writeHead(200, { 'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream' });
  res.end(readFileSync(file));
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const { port } = server.address();

const browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: ['--no-sandbox'] });
try {
  const page = await browser.newPage();
  await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }]);
  const html = readFileSync(join(DIST, 'resume.html'), 'utf8').replace(/<[^>]+>/g, ' ');
  if (/\+\d{2}[\s-]?\d{4,}|\b\d{5}[\s-]?\d{5}\b/.test(html)) throw new Error('Résumé looks like it contains a phone number.');

  const response = await page.goto(`http://127.0.0.1:${port}/resume`, { waitUntil: 'networkidle0' });
  if (!response?.ok()) throw new Error(`/resume returned ${response?.status()}`);
  await page.evaluate(() => document.fonts.ready);

  const pdf = await page.pdf({ preferCSSPageSize: true, printBackground: true, tagged: true, outline: true });
  const text = Buffer.from(pdf).toString('latin1');
  writeFileSync(join(DIST, 'resume.pdf'), pdf);
  const pages = (text.match(/\/Type\s*\/Page[^s]/g) ?? []).length;
  console.log(`build-resume-pdf: wrote dist/resume.pdf (${pages} page${pages === 1 ? '' : 's'}, ${Math.round(pdf.length / 1024)} KB)`);
} finally {
  await browser.close();
  server.close();
}
