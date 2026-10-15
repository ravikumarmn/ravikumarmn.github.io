#!/usr/bin/env node
/**
 * Lists every unresolved placeholder in the site content:
 *   TODO   facts the résumé doesn't provide (rendered as visible amber badges)
 *   VERIFY copy that was inferred and needs your confirmation (underlined in `npm run dev`)
 *
 * Usage: npm run audit:content
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const ROOT = process.cwd();
const DIRS = ['src/content/site', 'src/content/work', 'src/components'];
const files = [];
const walk = (dir) => {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) walk(path);
    else if (/\.(mdx?|json)$/.test(name)) files.push(path);
  }
};
DIRS.forEach((dir) => walk(join(ROOT, dir)));

const found = { TODO: [], VERIFY: [] };
for (const file of files) {
  readFileSync(file, 'utf8')
    .split('\n')
    .forEach((line, i) => {
      const where = `${relative(ROOT, file)}:${i + 1}`;
      for (const m of line.matchAll(/<Todo[^>]*>(.*?)<\/Todo>|"TODO:?\s*([^"]*)"|'TODO:?\s*([^']*)'/g)) {
        found.TODO.push(`${where}  ${(m[1] ?? m[2] ?? m[3]).trim()}`);
      }
      for (const m of line.matchAll(/<Verify[^>]*note="([^"]*)"/g)) found.VERIFY.push(`${where}  ${m[1]}`);
      if (/"verify":/.test(line)) found.VERIFY.push(`${where}  ${line.split('"verify":')[1].trim().replace(/^"|"$/g, '')}`);
    });
}

if (!existsSync(join(ROOT, 'public/resume.pdf'))) {
  found.TODO.unshift('public/resume.pdf  Add a résumé PDF without your phone number (the Résumé buttons show TODO until then).');
}

for (const [kind, items] of Object.entries(found)) {
  console.log(`\n${kind} (${items.length})`);
  items.forEach((item) => console.log(`  ${item}`));
}
console.log(
  found.TODO.length
    ? '\nResolve TODOs, then build with STRICT_CONTENT=1 to make sure none ship.'
    : '\nNo TODOs left. Consider setting STRICT_CONTENT=1 in the deploy workflow.',
);
