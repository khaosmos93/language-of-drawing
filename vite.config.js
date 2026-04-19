import { defineConfig } from "vite";

const repoBase = process.env.VITE_BASE_PATH || "/";

export default defineConfig({
  root: "web",
  // GitHub Pages project site: set VITE_BASE_PATH="/REPO_NAME/"
  // Root hosting or custom domain: keep VITE_BASE_PATH="/"
  base: repoBase,
  build: {
    outDir: "../dist",
    emptyOutDir: true,
  },
  server: {
    fs: {
      allow: [".."],
    },
  },
});
