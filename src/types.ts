export type Site = {
  NAME: string;
  EMAIL: string;
  NUM_POSTS_ON_HOMEPAGE: number;
  NUM_WORKS_ON_HOMEPAGE: number;
  NUM_PROJECTS_ON_HOMEPAGE: number;
};

export type Metadata = {
  TITLE: string;
  DESCRIPTION: string;
};

export type Socials = {
  NAME: string;
  HREF: string;
}[];

export type Presentation = {
  TITLE: string;
  VENUE: { en: string; ca: string };
  DATE: string;
  DESCRIPTION: string;
  SLIDES: string;
};
