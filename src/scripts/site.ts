/**
 * The site's only client JS. Each feature is progressive: without JS the page
 * is fully readable and navigable.
 */

const root = document.documentElement;
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

// Theme toggle. The initial theme is applied pre-paint in Base.astro.
function setTheme(theme: 'light' | 'dark', persist: boolean) {
  root.dataset.theme = theme;
  if (persist) {
    try {
      localStorage.setItem('theme', theme);
    } catch {}
  }
}
document.querySelectorAll<HTMLButtonElement>('[data-theme-toggle]').forEach((button) => {
  button.addEventListener('click', () => setTheme(root.dataset.theme === 'dark' ? 'light' : 'dark', true));
});
// Follow OS changes until the visitor picks a theme explicitly.
matchMedia('(prefers-color-scheme: light)').addEventListener('change', (event) => {
  let saved: string | null = null;
  try {
    saved = localStorage.getItem('theme');
  } catch {}
  if (!saved) setTheme(event.matches ? 'light' : 'dark', false);
});

// Header border/blur once the page has scrolled.
const header = document.querySelector<HTMLElement>('[data-header]');
if (header) {
  const onScroll = () => header.toggleAttribute('data-scrolled', window.scrollY > 8);
  onScroll();
  window.addEventListener('scroll', onScroll, { passive: true });
}

// Scroll-in reveal (CSS only hides elements when motion is allowed).
const revealables = document.querySelectorAll<HTMLElement>('[data-reveal]');
if (reduceMotion || !('IntersectionObserver' in window)) {
  revealables.forEach((el) => el.classList.add('is-in'));
} else {
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        entry.target.classList.add('is-in');
        observer.unobserve(entry.target);
      }
    },
    { rootMargin: '0px 0px -8% 0px', threshold: 0.05 },
  );
  revealables.forEach((el) => observer.observe(el));
}

// Highlight the nav link for the section in view (home page only).
const navLinks = document.querySelectorAll<HTMLAnchorElement>('[data-nav]');
const sections = [...navLinks]
  .map((link) => document.getElementById(link.dataset.nav!))
  .filter((section): section is HTMLElement => section !== null);
if (sections.length) {
  const visible = new Map<string, boolean>();
  const sectionObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) visible.set(entry.target.id, entry.isIntersecting);
      const current = sections.find((section) => visible.get(section.id))?.id;
      navLinks.forEach((link) => link.setAttribute('aria-current', String(link.dataset.nav === current)));
    },
    { rootMargin: '-40% 0px -55% 0px' },
  );
  sections.forEach((section) => sectionObserver.observe(section));
}

// Case-study jump bar: mark the section whose heading was most recently scrolled past.
const tocLinks = [...document.querySelectorAll<HTMLAnchorElement>('[data-toc] a')];
const tocTargets = tocLinks.map((link) => document.getElementById(decodeURIComponent(link.hash.slice(1))));
if (tocLinks.length) {
  const onScroll = () => {
    const line = window.innerHeight * 0.3;
    let current = -1;
    tocTargets.forEach((target, index) => {
      if (target && target.getBoundingClientRect().top < line) current = index;
    });
    tocLinks.forEach((link, index) => link.setAttribute('aria-current', String(index === current)));
  };
  onScroll();
  window.addEventListener('scroll', onScroll, { passive: true });
}

// Close the mobile menu after choosing a destination.
const mobileNav = document.querySelector<HTMLElement>('[data-mobile-nav]');
mobileNav?.querySelectorAll('a').forEach((link) => link.addEventListener('click', () => mobileNav.hidePopover?.()));

// Copy email to clipboard.
document.querySelectorAll<HTMLButtonElement>('[data-copy]').forEach((button) => {
  const label = button.querySelector<HTMLElement>('[data-copy-label]');
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(button.dataset.copy!);
      button.dataset.copied = '';
      if (label) label.textContent = 'Copied';
      setTimeout(() => {
        delete button.dataset.copied;
        if (label) label.textContent = 'Copy';
      }, 1800);
    } catch {
      window.location.href = `mailto:${button.dataset.copy}`;
    }
  });
});
