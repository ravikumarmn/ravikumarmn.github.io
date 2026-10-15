import { getCollection } from 'astro:content';

export async function getWork() {
  const entries = await getCollection('work', ({ data }) => !data.draft);
  return entries.sort((a, b) => a.data.order - b.data.order);
}

export async function getPosts() {
  const entries = await getCollection('posts');
  return entries.sort((a, b) => b.data.date.getTime() - a.data.date.getTime());
}

/** Minutes to read, at ~220 words per minute; code blocks are skimmed, so they count for less. */
export function readingTime(body = '') {
  const code = (body.match(/```[\s\S]*?```/g) ?? []).join(' ');
  const prose = body.replace(/```[\s\S]*?```/g, ' ');
  const words = (text: string) => text.split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.round((words(prose) + words(code) / 3) / 220));
}
