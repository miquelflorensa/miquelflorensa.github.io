export type Lang = "en" | "ca";

export const NAV_TEXT = {
  en: { blog: "blog", work: "work", projects: "projects", presentations: "presentations" },
  ca: { blog: "blog", work: "feina", projects: "projectes", presentations: "presentacions" },
};

export const PRESENTATIONS_TEXT = {
  en: {
    title: "Presentations",
    description: "Slides from my seminars and presentations.",
    dateLocale: "en-US",
  },
  ca: {
    title: "Presentacions",
    description: "Diapositives dels meus seminaris i presentacions.",
    dateLocale: "ca-ES",
  },
};

export const HOME_TEXT = {
  en: {
    description: "Miquel Florensa Montilla porfolio and blog.",
    hello: "Hello there! I'm Miquel",
    intro: [
      `I am a Ph.D. student at Polytechnique Montréal, working within the <a href="https://tagiml.com">BayesWorks</a> and <a href="https://www.polymtl.ca/litiv/en/research">LITIV</a> (Computer Vision) research groups. My primary research focuses on Diffusion Models integrated with Bayesian Neural Networks, combining probabilistic inference and deep learning to advance uncertainty quantification and generative modeling.`,
      "Beyond my academic work, I am deeply passionate about generative AI, computer vision, and robotics. I am constantly exploring how these fields can intersect to drive innovation in machine learning and its real-world applications.",
      "In my free time, I enjoy diving into the world of science fiction, which fuels my imagination and inspires me to think about the future of technology.",
    ],
    latestPosts: "Latest posts",
    allPosts: "See all posts",
    recentProjects: "Recent projects",
    allProjects: "See all projects",
    recentPresentations: "Recent presentations",
    allPresentations: "See all presentations",
    presentationsSlug: "presentations",
    education: "Education",
    educationItems: [
      {
        dates: "May 2025 – Present",
        title: "Polytechnique Montréal, Ph.D. in Computer Engineering",
        points: ["Bayesian neural network dynamics and probabilistic classification", "CUDA kernel development for cuTAGI library"],
      },
      {
        dates: "Jan 2024 – Apr 2025",
        title: "Polytechnique Montréal, M.A.Sc. in Computer Engineering",
        points: ["Mathematical formulation for uncertainty in Diffusion Models", "C++/CUDA/Python development for cuTAGI"],
      },
      {
        dates: "Sept 2019 – June 2023",
        title: "Universitat Politècnica de Catalunya, B.S. in Computer Engineering",
        points: ["Specialization in Computer Science"],
      },
    ],
    skills: "Skills & Tools",
    connect: "Let's Connect",
    connectText: "If you want to get in touch, reach out on social media or send me an email.",
  },
  ca: {
    description: "Portfoli i blog de Miquel Florensa Montilla.",
    hello: "Hola! Sóc en Miquel",
    intro: [
      `Sóc estudiant de doctorat a Polytechnique Montréal, on treballo dins dels grups de recerca <a href="https://tagiml.com">BayesWorks</a> i <a href="https://www.polymtl.ca/litiv/en/research">LITIV</a> (visió per computador). La meva recerca se centra en els models de difusió combinats amb xarxes neuronals bayesianes, unint la inferència probabilística i l'aprenentatge profund per millorar la quantificació de la incertesa i els models generatius.`,
      "Més enllà de la recerca, m'apassionen la IA generativa, la visió per computador i la robòtica, i m'agrada explorar com aquests camps es poden combinar per impulsar l'aprenentatge automàtic i les seves aplicacions al món real.",
      "En el meu temps lliure m'agrada submergir-me en la ciència-ficció, que m'alimenta la imaginació i em fa pensar en el futur de la tecnologia.",
    ],
    latestPosts: "Últims articles",
    allPosts: "Tots els articles (en anglès)",
    recentProjects: "Projectes recents",
    allProjects: "Tots els projectes",
    recentPresentations: "Presentacions recents",
    allPresentations: "Totes les presentacions",
    presentationsSlug: "presentacions",
    education: "Formació",
    educationItems: [
      {
        dates: "maig 2025 – actualitat",
        title: "Polytechnique Montréal, doctorat en Enginyeria Informàtica",
        points: ["Dinàmica de les xarxes neuronals bayesianes i classificació probabilística", "Desenvolupament de kernels CUDA per a la llibreria cuTAGI"],
      },
      {
        dates: "gener 2024 – abril 2025",
        title: "Polytechnique Montréal, màster de recerca (M.A.Sc.) en Enginyeria Informàtica",
        points: ["Formulació matemàtica de la incertesa en els models de difusió", "Desenvolupament en C++/CUDA/Python per a cuTAGI"],
      },
      {
        dates: "setembre 2019 – juny 2023",
        title: "Universitat Politècnica de Catalunya, grau en Enginyeria Informàtica",
        points: ["Especialitat en Computació"],
      },
    ],
    skills: "Habilitats i eines",
    connect: "Contacte",
    connectText: "Si vols posar-te en contacte amb mi, escriu-me per xarxes socials o envia'm un correu.",
  },
};
