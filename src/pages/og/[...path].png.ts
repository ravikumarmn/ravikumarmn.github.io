/**
 * Open Graph images (1200×630), rendered at build time with satori + resvg:
 *   /og/home.png, /og/work/<slug>.png, /og/blogs/<slug>.png
 *
 * Built for how they are actually seen: shrunk to ~500px wide in a LinkedIn or X
 * feed. So type is large, labels use the muted (not subtle) tier, and nothing
 * important sits near the edges where some clients crop.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { APIRoute, GetStaticPaths } from 'astro';
import { Resvg } from '@resvg/resvg-js';
import satori from 'satori';
import { getPosts, getWork, readingTime } from '~/lib/collections';
import { metrics, profile } from '~/lib/site';

const font = (pkg: string, file: string) => readFileSync(join(process.cwd(), 'node_modules/@fontsource', pkg, 'files', file));
const fonts = [
  { name: 'Geist', data: font('geist', 'geist-latin-400-normal.woff'), weight: 400 as const },
  { name: 'Geist', data: font('geist', 'geist-latin-600-normal.woff'), weight: 600 as const },
  { name: 'Geist Mono', data: font('geist-mono', 'geist-mono-latin-500-normal.woff'), weight: 500 as const },
];

// Dark-theme tokens (src/styles/tokens.css).
const C = {
  bg: '#0b0c0e',
  surface: '#16181c',
  surface3: '#1d2024',
  text: '#edeef0',
  muted: '#a3a8b0',
  subtle: '#8a8f98',
  border: '#25282d',
  borderStrong: '#34383f',
  accent: '#5fd4c0',
};

interface Card {
  eyebrow: string;
  title: string;
  stats: { value: string; label: string }[];
  meta?: string;
}

const isTodo = (value: string) => value.trimStart().startsWith('TODO');
const ascii = (value: string) => value.replace(/−/g, '-').replace(/→/g, ' to ');

export const getStaticPaths = (async () => {
  const work = await getWork();
  const posts = await getPosts();
  return [
    {
      params: { path: 'home' },
      props: {
        eyebrow: profile.availability ?? profile.title,
        title: profile.headline,
        stats: metrics.slice(0, 3).map(({ value, label }) => ({ value, label })),
      },
    },
    ...work.map((entry) => ({
      params: { path: `work/${entry.id}` },
      props: {
        eyebrow: ['Case study', isTodo(entry.data.org) ? '' : entry.data.org].filter(Boolean).join(' · '),
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
        meta: [
          post.data.date.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric', timeZone: 'UTC' }),
          `${readingTime(post.body)} min read`,
        ].join('  ·  '),
      },
    })),
  ];
}) satisfies GetStaticPaths;

const h = (type: string, style: Record<string, unknown>, children?: unknown) => ({ type, props: { style, children } });

export const GET: APIRoute = async ({ props }) => {
  const { eyebrow, title, stats, meta } = props as Card;
  // Sized so every title lands on at most two lines at 1040px.
  const titleSize = title.length > 80 ? 52 : title.length > 60 ? 58 : title.length > 40 ? 66 : 76;

  const identity = h('div', { display: 'flex', alignItems: 'center', gap: 18 }, [
    h(
      'div',
      {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: 52,
        height: 52,
        borderRadius: 11,
        border: `1.5px solid ${C.borderStrong}`,
        backgroundImage: `linear-gradient(to bottom, ${C.surface3}, ${C.surface})`,
        fontFamily: 'Geist Mono',
        fontSize: 19,
        color: C.text,
        letterSpacing: -0.5,
      },
      profile.initials,
    ),
    h('div', { display: 'flex', flexDirection: 'column', gap: 2 }, [
      h('div', { display: 'flex', fontSize: 25, fontWeight: 600, color: C.text, letterSpacing: -0.4 }, profile.name),
      h('div', { display: 'flex', fontSize: 20, color: C.muted }, profile.title),
    ]),
  ]);

  const header = h('div', { display: 'flex', alignItems: 'center', justifyContent: 'space-between' }, [
    identity,
    h('div', { display: 'flex', fontFamily: 'Geist Mono', fontSize: 20, color: C.subtle }, 'ravikumarmn.github.io'),
  ]);

  const body = h('div', { display: 'flex', flexDirection: 'column', gap: 22 }, [
    h('div', { display: 'flex', alignItems: 'center', gap: 14, fontFamily: 'Geist Mono', fontSize: 20, color: C.accent, letterSpacing: 1.5, textTransform: 'uppercase' }, [
      h('div', { width: 10, height: 10, borderRadius: 999, backgroundColor: C.accent }),
      h('div', { display: 'flex' }, eyebrow),
    ]),
    h(
      'div',
      { display: 'flex', fontSize: titleSize, fontWeight: 600, lineHeight: 1.06, letterSpacing: -titleSize * 0.032, maxWidth: 1040, color: C.text },
      title,
    ),
  ]);

  const footer = stats.length
    ? h(
        'div',
        { display: 'flex', gap: 64, borderTop: `1.5px solid ${C.border}`, paddingTop: 30 },
        stats.map((stat) =>
          h('div', { display: 'flex', flexDirection: 'column', gap: 6, flexShrink: 0, maxWidth: 320 }, [
            h('div', { display: 'flex', fontFamily: 'Geist Mono', fontSize: 44, color: C.text, letterSpacing: -1.5 }, ascii(stat.value)),
            h('div', { display: 'flex', fontSize: 21, color: C.muted }, stat.label),
          ]),
        ),
      )
    : h(
        'div',
        { display: 'flex', borderTop: `1.5px solid ${C.border}`, paddingTop: 30, fontFamily: 'Geist Mono', fontSize: 22, color: C.muted },
        meta ?? '',
      );

  const tree = h(
    'div',
    {
      position: 'relative',
      display: 'flex',
      width: 1200,
      height: 630,
      backgroundColor: C.bg,
      backgroundImage: `radial-gradient(circle at 8% 0%, rgba(95,212,192,0.17), transparent 55%)`,
      color: C.text,
      fontFamily: 'Geist',
    },
    [
      // A hairline of accent along the top edge: the one brand mark on the card.
      h('div', {
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: 3,
        backgroundImage: `linear-gradient(to right, ${C.accent}, rgba(95,212,192,0) 70%)`,
      }),
      h(
        'div',
        {
          position: 'absolute',
          top: 60,
          right: 72,
          bottom: 60,
          left: 72,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
        },
        [header, body, footer],
      ),
    ],
  );

  const svg = await satori(tree as Parameters<typeof satori>[0], { width: 1200, height: 630, fonts });
  const png = new Resvg(svg, { fitTo: { mode: 'width', value: 1200 } }).render().asPng();
  return new Response(new Uint8Array(png), { headers: { 'Content-Type': 'image/png' } });
};
