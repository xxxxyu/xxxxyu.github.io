# xxxxyu.github.io

Personal academic homepage and blog for Xiangyu Li, built with [Zola](https://www.getzola.org/).

## Development

```bash
# Install Zola (macOS)
brew install zola

# Local dev server with live reload
zola serve

# Production build
zola build
```

The site builds to `public/`.

## Deployment

Pushes to `main` automatically deploy to GitHub Pages via the workflow in `.github/workflows/deploy.yml`.

To enable: go to repo Settings > Pages > Source > GitHub Actions.

## Structure

```
content/          # Markdown content (homepage, about, blog posts)
data/             # Structured data (publications.toml)
templates/        # Zola Tera templates
sass/             # SCSS stylesheets
static/           # Static assets (images, fonts, files)
```

## Credits

- [Zola](https://www.getzola.org/) — static site generator
- Design inspired by [Minimal Light](https://github.com/yaoyao-liu/minimal-light) (academic layout) and [Tabi](https://github.com/welpo/tabi) (blog layout)
