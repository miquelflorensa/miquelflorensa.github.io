# miquelflorensa.github.io

> 👋 Hi, I’m Miquel Florensa—Ph.D. student in Computer Engineering at Polytechnique Montréal  
> This is my personal website built with [Astro 3.0](https://astro.build) and deployed via GitHub Pages at:  
> `https://miquelflorensa.github.io/`

---

## 🚀 About Me

I’m currently pursuing my Ph.D. at Polytechnique Montréal, working with the BayesWorks and LITIV research groups.  
My research combines **Diffusion Models** with **Bayesian Neural Networks** to push the boundaries of uncertainty quantification and generative modeling.  

- 🔬 **Research interests:**  
  - Probabilistic deep learning & uncertainty  
  - Generative models (diffusion, autoregressive)  
  - CUDA kernel development for high‑performance inference  
- 💡 **Passion projects:**  
  - Generative AI demos  
  - Computer vision pipelines  
  - Robotics & autonomous systems  
- 📚 **When I’m offline:**  
  - Sci‑fi reading & world‑building  
  - Open‑source contributions  

---

## 🏗️ Tech Stack

- **Framework:** Astro 3.0  
- **Languages:** TypeScript / JavaScript / Astro  
- **Styling:** Tailwind CSS  
- **Content:** `astro:content` collections for blog, projects, and CV  
- **Components & Layouts:**  
  - `Container.astro`, `PageLayout.astro`  
  - Custom cards (`ArrowCard.astro`), links, and utility components  
- **Utilities:**  
  - `@lib/utils` for date formatting, constants  
  - Environment constants in `@consts` (site URL, social links, etc.)  
- **Deployment:** GitHub Pages via Actions

---

## 📥 Getting Started

1. **Clone this repo**  
   ```bash
   git clone https://github.com/miquelflorensa/miquelflorensa.github.io.git
   cd miquelflorensa.github.io
   ```

2. **Install dependencies**  
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Configure your environment**  
   - All settings (site URL, number of posts/projects to show, socials, etc.) live in `src/consts.ts`.  
   - Update any personal links or email in that file.

4. **Run development server**  
   ```bash
   npm run dev
   # or
   yarn dev
   ```
   Preview at <http://localhost:3000>.

---

## 📦 Build & Deploy

### Local build
```bash
npm run build
# or
yarn build
```
- Outputs static files to `dist/`.

### GitHub Actions (auto‑deploy)
A workflow in `.github/workflows/deploy.yml` will:
1. Install dependencies  
2. Build the site  
3. Publish the `dist/` folder to GitHub Pages (root of this repo’s `main` branch)

> **Note:** You do *not* commit `node_modules/`; dependencies are installed on the runner.

---

## 📂 Repository Structure

```text
/
├── .github/
│   └── workflows/       # GitHub Actions for build & deploy
├── public/              # Static assets (images, icons, robots.txt)
├── src/
│   ├── components/      # Reusable Astro/Vue/React components
│   ├── layouts/         # Page layout templates
│   ├── pages/           # Top‑level routes (index.astro, blog.astro, etc.)
│   ├── content/         # Markdown/MDX collections: blog, projects, work
│   └── lib/             # Utility functions (date formatting, constants)
├── astro.config.mjs     # Astro configuration (site URL, base path)
├── package.json
├── tsconfig.json        # TypeScript settings
└── README.md            # ← this file
```

---

## 📫 Connect with Me

- 📧 **Email:** miquelflorensa@polymtl.ca  
- 💼 **LinkedIn:** [linkedin.com/in/miquelflorensa](https://linkedin.com/in/miquelflorensa)  
- 🐦 **Twitter/X:** [@MiquelFlorensa](https://twitter.com/MiquelFlorensa)  
- 📂 **GitHub:** [github.com/miquelflorensa](https://github.com/miquelflorensa)

---

*Built with ❤️ using Astro.*  