/**
 * Typed access to the site content in src/content/site/*.json.
 * Each file is validated at build time, so a typo in the JSON fails the build
 * with a readable error instead of rendering a broken page.
 */
import { z } from 'astro/zod';
import profileJson from '../content/site/profile.json';
import metricsJson from '../content/site/metrics.json';
import experienceJson from '../content/site/experience.json';
import skillsJson from '../content/site/skills.json';
import projectsJson from '../content/site/projects.json';
import aboutJson from '../content/site/about.json';
import buildingJson from '../content/site/building.json';

const IconName = z.enum(['github', 'linkedin', 'mail']);

const Profile = z.object({
  name: z.string(),
  shortName: z.string(),
  initials: z.string(),
  title: z.string(),
  company: z.string(),
  location: z.string(),
  availability: z.string().optional(),
  email: z.string(),
  resume: z.string(),
  headline: z.string(),
  subline: z.string(),
  seoDescription: z.string(),
  links: z.array(z.object({ label: z.string(), href: z.string(), icon: IconName, handle: z.string() })),
  education: z.object({ degree: z.string(), school: z.string(), period: z.string(), note: z.string().optional() }),
  certifications: z.array(z.string()),
});

const Metric = z.object({
  value: z.string(),
  label: z.string(),
  context: z.string(),
  /** Where the number comes from, so every claim on the page stays traceable. */
  source: z.string(),
});

const Role = z.object({
  role: z.string(),
  org: z.string(),
  type: z.string(),
  location: z.string(),
  start: z.string(),
  end: z.string(),
  summary: z.string(),
  highlights: z.array(z.string()).min(1).max(3),
  stack: z.array(z.string()),
});

const SkillGroup = z.object({ group: z.string(), items: z.array(z.string()) });

const Project = z.object({
  title: z.string(),
  detail: z.string(),
  metric: z.string(),
  stack: z.string(),
  href: z.string().optional(),
});

const About = z.object({ paragraphs: z.array(z.string()), verify: z.string().optional() });

/** Projects in progress, each tied to real experience. No results are claimed until they are measured. */
const BuildingProject = z.object({
  slug: z.string(),
  title: z.string(),
  summary: z.string(),
  /** The real work this project builds on, linked to its case study. */
  buildsOn: z.object({ text: z.string(), href: z.string() }),
  approach: z.array(z.string()).min(1),
  measures: z.array(z.string()).min(1),
  publishes: z.array(z.string()).min(1),
  stack: z.array(z.string()),
});

export const profile = Profile.parse(profileJson);
export const metrics = z.array(Metric).parse(metricsJson);
export const experience = z.array(Role).parse(experienceJson);
export const skills = z.array(SkillGroup).parse(skillsJson);
export const projects = z.array(Project).parse(projectsJson);
export const about = About.parse(aboutJson);
export const building = z.array(BuildingProject).parse(buildingJson);

export type IconName = z.infer<typeof IconName>;

/** Values that start with "TODO" are rendered as visible placeholders. */
export const isTodo = (value: string) => value.trimStart().startsWith('TODO');
export const todoText = (value: string) => value.replace(/^\s*TODO:?\s*/, '');


/**
 * `id` links scroll to a homepage section (highlighted while it is in view);
 * `page` links go to their own page (highlighted on any path they match).
 */
export const navLinks: { href: string; label: string; id?: string; page?: RegExp }[] = [
  { href: '/#work', label: 'Work', id: 'work' },
  { href: '/#projects', label: 'Projects', id: 'projects', page: /^\/projects$/ },
  { href: '/#experience', label: 'Experience', id: 'experience' },
  { href: '/writing', label: 'Writing', page: /^\/(writing|blogs)(\/|$)/ },
  { href: '/#about', label: 'About', id: 'about' },
  { href: '/#contact', label: 'Contact', id: 'contact' },
];
