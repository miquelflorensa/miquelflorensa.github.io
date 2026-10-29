import type { Site, Metadata, Socials, Presentation } from "@types";

export const SITE: Site = {
  NAME: "Miquel Florensa Montilla",
  EMAIL: "miquelflorensa11@gmail.com",
  NUM_POSTS_ON_HOMEPAGE: 3,
  NUM_WORKS_ON_HOMEPAGE: 2,
  NUM_PROJECTS_ON_HOMEPAGE: 3,
};

export const HOME: Metadata = {
  TITLE: "Miquel Florensa Montilla",
  DESCRIPTION: "Miquel Florensa Montilla porfolio and blog.",
};

export const BLOG: Metadata = {
  TITLE: "Blog",
  DESCRIPTION: "A collection of articles on topics I am passionate about.",
};

export const WORK: Metadata = {
  TITLE: "Work",
  DESCRIPTION: "Where I have worked and what I have done.",
};

export const PROJECTS: Metadata = {
  TITLE: "Projects",
  DESCRIPTION: "A collection of my projects, with links to repositories and demos.",
};

export const PRESENTATIONS: Presentation[] = [
  {
    TITLE: "Cap-free Batch Update for TAGI",
    VENUE: { en: "Seminar, Polytechnique Montréal", ca: "Seminari, Polytechnique Montréal" },
    DATE: "2026-10-29",
    DESCRIPTION: "One step length, measured from the step itself.",
    SLIDES: "/presentations/capfree-tagi/capfree_talk.html",
  },
];

export const SOCIALS: Socials = [
  { 
    NAME: "x",
    HREF: "https://x.com/maikk_11",
  },
  { 
    NAME: "github",
    HREF: "https://github.com/miquelflorensa"
  },
  { 
    NAME: "linkedin",
    HREF: "https://www.linkedin.com/in/miquel-florensa/",
  }
];
