#!/usr/bin/env node
/**
 * One-off migration: Jekyll/Chirpy posts in _posts/ → Astro content in src/content/posts/.
 *
 * - Keeps each post's slug exactly as Jekyll built it (/blogs/:title), so old URLs keep working.
 * - Copies referenced images into src/assets/posts/ (optimized by Astro) with URL-safe names;
 *   animated GIFs go to public/images/posts/ untouched.
 * - Converts <img> HTML to Markdown images, Liquid includes to lazy iframes, and strips the
 *   old MathJax and polyfill.io scripts (KaTeX renders math at build time now).
 *
 * Already run. The Jekyll sources it reads (_posts/, images/, _includes/) were removed afterwards
 * and live in git history; restore them first if you ever need to re-run it.
 * Usage: node scripts/migrate-posts.mjs
 */
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, dirname, join } from 'node:path';
import sharp from 'sharp';
import { parse } from 'yaml';

const ROOT = process.cwd();
const SRC = join(ROOT, '_posts');
const OUT = join(ROOT, 'src/content/posts');
const ASSETS = join(ROOT, 'src/assets/posts');
const PUBLIC_IMAGES = join(ROOT, 'public/images/posts');
const EMBEDS = join(ROOT, 'public/embeds');

const FEATURED = new Set([
  'retrieval-augmented-generation-(rag)',
  'search-neurips-2022-similar-papers',
  'concepts-you-should-know-before-getting-into-transformer',
  'multi-label-classification-using-huggingface-transformer',
]);

/** Meta descriptions where the post's first paragraph doesn't make a usable one. */
const DESCRIPTIONS = {
  'sentiment-classification-and-word-embedding':
    'Sentiment classification on IMDB movie reviews: data exploration, word embeddings, an LSTM classifier, and t-SNE views of what the embeddings learn.',
  'pytorch-functions-you-should-know,-before-building-transformer':
    'The PyTorch tensor operations you need before implementing a transformer from scratch, each with a short example.',
  'understanding-linear-regression-in-supervised-learning':
    'Linear regression from first principles: the model, the cost function, and gradient descent, with an interactive plot and a from-scratch notebook.',
};

const EMBED_TITLES = {
  'interactives/Interactive_Linear_Regression.html': ['Interactive linear regression (Plotly)', 640],
  'notebooks/linear_regression.html': ['Linear regression notebook', 900],
};

const safeName = (path) =>
  path
    .split('/')
    .map((part) => part.toLowerCase().replace(/[^a-z0-9._-]+/g, '-'))
    .join('/');

const missing = [];

function migrateImage(src, slug) {
  const rel = decodeURI(src).replace(/^\/?images\//, '');
  const from = join(ROOT, 'images', rel);
  if (!existsSync(from)) {
    missing.push(`${slug}: images/${rel}`);
    return null;
  }
  const target = safeName(rel);
  if (from.toLowerCase().endsWith('.gif')) {
    mkdirSync(dirname(join(PUBLIC_IMAGES, target)), { recursive: true });
    copyFileSync(from, join(PUBLIC_IMAGES, target));
    return `/images/posts/${target}`;
  }
  mkdirSync(dirname(join(ASSETS, target)), { recursive: true });
  copyFileSync(from, join(ASSETS, target));
  return `../../assets/posts/${target}`;
}

/** Applies a transform to everything outside fenced code blocks. */
function outsideCode(body, transform) {
  return body
    .split(/(^```[\s\S]*?^```)/m)
    .map((part, i) => (i % 2 === 1 ? part : transform(part)))
    .join('');
}

function firstParagraph(body) {
  const para = body
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .find((p) => p && !/^(#|!\[|<|```|\||-|\*|>)/.test(p));
  if (!para) return undefined;
  const text = para
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[`*_]/g, '')
    .replace(/\$\$/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  if (text.length <= 158) return text;
  // Prefer whole sentences; fall back to a word boundary.
  let out = '';
  for (const sentence of text.match(/[^.!?]+[.!?]+/g) ?? []) {
    if ((out + sentence).trim().length > 158) break;
    out += sentence;
  }
  return out.trim() || `${text.slice(0, 155).replace(/\s+\S*$/, '')}…`;
}

mkdirSync(OUT, { recursive: true });
mkdirSync(EMBEDS, { recursive: true });

for (const file of readdirSync(SRC).filter((f) => f.endsWith('.md'))) {
  const raw = readFileSync(join(SRC, file), 'utf8');
  const match = raw.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) throw new Error(`No front matter in ${file}`);
  const fm = parse(match[1]);
  const slug = basename(file, '.md').replace(/^\d{4}-\d{2}-\d{2}-/, '');

  let body = outsideCode(match[2], (text) => {
    let out = text;
    // Scripts the old theme needed (MathJax, and polyfill.io, which was compromised in 2024).
    out = out.replace(/<script[^>]*(polyfill\.io|mathjax)[^>]*><\/script>\s*/gi, '');
    // <div ...><img ...></div> and bare <img> → Markdown image.
    out = out.replace(/<div[^>]*>\s*(<img[^>]*>)\s*<\/div>/gi, '$1');
    out = out.replace(/<img([^>]*)>/gi, (_, attrs) => {
      const src = attrs.match(/src="([^"]+)"/)?.[1];
      const alt = attrs.match(/alt="([^"]*)"/)?.[1] ?? '';
      return src ? `![${alt}](${src})` : '';
    });
    // Markdown images under images/ → migrated assets. Paths may contain spaces.
    out = out.replace(/!\[([^\]]*)\]\((\/?images\/[^)]+?)(\s+"[^"]*")?\)/g, (all, alt, src) => {
      const next = migrateImage(src.trim(), slug);
      return next ? `![${alt}](${next})` : all;
    });
    // Liquid includes → lazy iframes of the original standalone HTML.
    out = out.replace(/\{%\s*include\s+(\S+)\s*%\}/g, (_, path) => {
      const [title, height] = EMBED_TITLES[path] ?? [path, 600];
      const name = basename(path);
      copyFileSync(join(ROOT, '_includes', path), join(EMBEDS, name));
      return `<iframe src="/embeds/${name}" title="${title}" loading="lazy" height="${height}"></iframe>`;
    });
    // Angle-bracket tokens that browsers would parse as tags.
    out = out.replace(/<(BOS|EOS)>/g, '`<$1>`').replace(/<class '([^']+)'>/g, "`<class '$1'>`");
    // Deprecated <center><font> caption markup.
    out = out.replace(/<center>\s*<font[^>]*>([\s\S]*?)<\/font>\s*<\/center>/gi, '<p class="caption">$1</p>');
    // The page title is the only <h1>; demote in-body h1s and drop bold-wrapped headings.
    out = out.replace(/^# (?=\S)/gm, '## ');
    out = out.replace(/^(#{2,6}) \*\*(.+?)\*\*\s*$/gm, '$1 $2');
    return out;
  });
  // Animated GIFs bypass Astro's optimizer, so give them explicit dimensions (no layout shift).
  for (const [all, alt, src] of body.matchAll(/!\[([^\]]*)\]\((\/images\/posts\/[^)]+\.gif)\)/g)) {
    const { width, height } = await sharp(join(ROOT, 'public', src)).metadata();
    body = body.replace(
      all,
      `<img src="${src}" alt="${alt.replace(/"/g, '&quot;')}" width="${width}" height="${height}" loading="lazy" decoding="async" />`,
    );
  }
  body = body.replace(/\n{3,}/g, '\n\n').trim() + '\n';

  const categories = fm.categories ?? [];
  const frontmatter = {
    title: fm.title,
    slug,
    date: new Date(fm.date).toISOString(),
    description: DESCRIPTIONS[slug] ?? firstParagraph(body),
    topic: categories.at(-1) ?? categories[0] ?? 'Information Retrieval',
    tags: fm.tags ?? [],
    featured: FEATURED.has(slug),
    math: /\$\$/.test(body),
  };
  const yaml = Object.entries(frontmatter)
    .filter(([, v]) => v !== undefined)
    .map(([k, v]) => `${k}: ${JSON.stringify(v)}`)
    .join('\n');

  writeFileSync(join(OUT, `${slug.replace(/[^a-z0-9-]+/g, '-').replace(/-+/g, '-').replace(/-$/, '')}.md`), `---\n${yaml}\n---\n\n${body}`);
  console.log(`✓ /blogs/${slug}`);
}

if (missing.length) {
  console.warn(`\nMissing images (left as-is):\n  ${missing.join('\n  ')}`);
  process.exitCode = 1;
}
