import { profile } from './site';

const SITE = import.meta.env.SITE.replace(/\/$/, '');
const PERSON_ID = `${SITE}/#person`;
const WEBSITE_ID = `${SITE}/#website`;

/** Home page: the person and the site, as one graph. */
export function personJsonLd() {
  return {
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'Person',
        '@id': PERSON_ID,
        name: profile.name,
        alternateName: 'Ravikumar M. N.',
        url: SITE,
        email: `mailto:${profile.email}`,
        jobTitle: profile.title,
        description: profile.seoDescription,
        worksFor: { '@type': 'Organization', name: profile.company },
        alumniOf: { '@type': 'CollegeOrUniversity', name: profile.education.school },
        knowsAbout: ['Multi-agent systems', 'Retrieval-augmented generation', 'LLM fine-tuning', 'Document AI', 'MLOps'],
        sameAs: profile.links.map((link) => link.href),
      },
      {
        '@type': 'WebSite',
        '@id': WEBSITE_ID,
        url: SITE,
        name: profile.name,
        description: profile.seoDescription,
        inLanguage: 'en',
        author: { '@id': PERSON_ID },
        publisher: { '@id': PERSON_ID },
      },
    ],
  };
}

export function articleJsonLd(input: { title: string; description?: string; url: URL; date?: Date; image?: URL }) {
  return {
    '@context': 'https://schema.org',
    '@type': 'Article',
    headline: input.title,
    description: input.description,
    url: input.url.href,
    mainEntityOfPage: input.url.href,
    image: input.image?.href,
    datePublished: input.date?.toISOString(),
    inLanguage: 'en',
    author: { '@id': PERSON_ID, '@type': 'Person', name: profile.name, url: SITE },
    publisher: { '@id': PERSON_ID },
    isPartOf: { '@id': WEBSITE_ID, '@type': 'WebSite', url: SITE, name: profile.name },
  };
}
