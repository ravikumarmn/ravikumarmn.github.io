import { getCollection } from 'astro:content';

export async function getWork() {
  const entries = await getCollection('work', ({ data }) => !data.draft);
  return entries.sort((a, b) => a.data.order - b.data.order);
}

export async function getPosts() {
  const entries = await getCollection('posts');
  return entries.sort((a, b) => b.data.date.getTime() - a.data.date.getTime());
}
