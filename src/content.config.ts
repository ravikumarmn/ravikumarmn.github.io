import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

/** Case studies: src/content/work/*.mdx */
const work = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/work' }),
  schema: z.object({
    title: z.string(),
    /** One line shown on the homepage card: the problem, not the tech. */
    summary: z.string(),
    /** Omitted when the résumé doesn't state it; the page leaves that field out rather than guessing. */
    role: z.string().optional(),
    org: z.string(),
    period: z.string().optional(),
    order: z.number(),
    /** Headline numbers. The first two appear on the homepage card. */
    metrics: z.array(z.object({ value: z.string(), label: z.string() })).min(2),
    stack: z.array(z.string()),
    links: z.array(z.object({ label: z.string(), href: z.string() })).default([]),
    draft: z.boolean().default(false),
  }),
});

/** Blog posts migrated from the old Jekyll site: src/content/posts/*.md */
const posts = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/posts' }),
  schema: ({ image }) =>
    z.object({
      title: z.string(),
      date: z.coerce.date(),
      description: z.string().optional(),
      topic: z.string(),
      tags: z.array(z.string()).default([]),
      cover: z.object({ src: image(), alt: z.string() }).optional(),
      /** Shown in the homepage Writing list. */
      featured: z.boolean().default(false),
      /** Pages that embed interactive HTML or math need extra assets. */
      math: z.boolean().default(false),
    }),
});

export const collections = { work, posts };
