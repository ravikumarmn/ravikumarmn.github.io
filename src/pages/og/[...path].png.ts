/**
 * Open Graph images (1200×630), rendered at build time with satori + resvg:
 *   /og/home.png, /og/work/<slug>.png, /og/blogs/<slug>.png
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { APIRoute, GetStaticPaths } from 'astro';
import { Resvg } from '@resvg/resvg-js';
import satori from 'satori';
import { getPosts, getWork } from '~/lib/collections';
import { metrics, profile } from '~/lib/site';

const font = (pkg: string, file: string) => readFileSync(join(process.cwd(), 'node_modules/@fontsource', pkg, 'files', file));
const fonts = [
  { name: 'Geist', data: font('geist', 'geist-latin-400-normal.woff'), weight: 400 as const },
  { name: 'Geist', data: font('geist', 'geist-latin-600-normal.woff'), weight: 600 as const },
  { name: 'Geist Mono', data: font('geist-mono', 'geist-mono-latin-500-normal.woff'), weight: 500 as const },
];

const C = { bg: '#0b0c0e', text: '#edeef0', muted: '#a1a6ae', subtle: '#868b94', border: '#23262b', accent: '#5fd4c0' };

interface Card {
  eyebrow: string;
  title: string;
  stats: { value: string; label: string }[];
}

export const getStaticPaths = (async () => {
  const work = await getWork();
  const posts = await getPosts();
  const headline = (value: string) => (value.startsWith('TODO') ? '' : value);
  return [
    {
      params: { path: 'home' },
      props: { eyebrow: profile.title, title: profile.headline, stats: metrics.slice(0, 3).map(({ value, label }) => ({ value, label })) },
    },
    ...work.map((entry) => ({
      params: { path: `work/${entry.id}` },
      props: {
        eyebrow: ['Case study', headline(entry.data.org)].filter(Boolean).join(' · '),
        title: entry.data.title,
        stats: entry.data.metrics.slice(0, 3),
      },
    })),
    ...posts.map((post) => ({
      params: { path: `blogs/${post.id}` },
      props: {
        eyebrow: `Writing · ${post.data.topic}`,
        title: post.data.title,
        stats: [],
      },
    })),
  ];
}) satisfies GetStaticPaths;

const h = (type: string, style: Record<string, unknown>, children?: unknown) => ({ type, props: { style, children } });

export const GET: APIRoute = async ({ props }) => {
  const { eyebrow, title, stats } = props as Card;
  const titleSize = title.length > 70 ? 52 : title.length > 44 ? 60 : 68;

  const content = h(
    'div',
    {
      position: 'absolute',
      top: 64,
      right: 72,
      bottom: 56,
      left: 72,
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
    },
    [
      h('div', { display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontFamily: 'Geist Mono', fontSize: 20, color: C.subtle }, [
        h('div', { display: 'flex', alignItems: 'center', gap: 14 }, [
          h('div', { width: 12, height: 12, borderRadius: 999, backgroundColor: C.accent }),
          h('div', { display: 'flex', textTransform: 'uppercase', letterSpacing: 2 }, `${profile.name}  ·  ${eyebrow}`),
        ]),
        h('div', { display: 'flex' }, 'ravikumarmn.github.io'),
      ]),
      h(
        'div',
        { display: 'flex', fontSize: titleSize, fontWeight: 600, lineHeight: 1.08, letterSpacing: -2, maxWidth: 1000 },
        title,
      ),
      h(
        'div',
        { display: 'flex', gap: 56, minHeight: 92, borderTop: `1px solid ${C.border}`, paddingTop: 28 },
        stats.map((stat) =>
          h('div', { display: 'flex', flexDirection: 'column', gap: 8, flexShrink: 0 }, [
            h('div', { display: 'flex', fontFamily: 'Geist Mono', fontSize: 36, color: C.text, letterSpacing: -1 }, stat.value.replace(/−/g, '-').replace(/→/g, ' to ')),
            h('div', { display: 'flex', fontSize: 19, color: C.subtle }, stat.label),
          ]),
        ),
      ),
    ],
  );
  const tree = h(
    'div',
    {
      position: 'relative',
      display: 'flex',
      width: 1200,
      height: 630,
      backgroundColor: C.bg,
      backgroundImage: `radial-gradient(circle at 12% 0%, rgba(95,212,192,0.16), transparent 52%)`,
      color: C.text,
      fontFamily: 'Geist',
    },
    [content],
  );

  const svg = await satori(tree as Parameters<typeof satori>[0], { width: 1200, height: 630, fonts });
  const png = new Resvg(svg, { fitTo: { mode: 'width', value: 1200 } }).render().asPng();
  return new Response(new Uint8Array(png), { headers: { 'Content-Type': 'image/png' } });
};
