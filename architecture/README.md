# Pydantic AI Architecture

This directory contains the LikeC4 source model and static-site toolchain for the Pydantic AI architecture viewer. The model files are linked from this [maintained branch](https://github.com/dsfaccini/pydantic-ai/tree/codex/likec4-architecture-site/architecture/model). Their source links pin the Pydantic AI code snapshot to revision [`001ed141fd1057d0035557353f87559eced5b0a7`](https://github.com/pydantic/pydantic-ai/tree/001ed141fd1057d0035557353f87559eced5b0a7).

## Work on the model

Edit the `.c4` files in `model/` to change architecture elements, relationships, and focused views. Keep each view scoped to a question it helps answer. The landing page opens the package map; the navigation menu lists the remaining views.

Keep `landingPage.redirect` enabled to load one diagram on entry. The preview grid retains diagrams as they enter the viewport, increasing browser rendering work as visitors scroll.

Install the pinned LikeC4 dependency with `pnpm install --frozen-lockfile`, then run `pnpm dev` to edit the model with the live viewer. Run `pnpm validate` before building to check the model.

## Build and preview

`pnpm build` generates the static viewer in `dist/`, with root-domain URLs. `pnpm preview` serves that output locally through Wrangler. Run `pnpm build` again after model changes before previewing.

To update the architecture snapshot after bringing this worktree up to date with Pydantic AI, run `git fetch upstream main` and `git merge upstream/main` here. Reconcile the model with the merged code and update its pinned source revision and component links; rebuilding alone does not update the snapshot. Then run `pnpm install --frozen-lockfile`, `pnpm validate`, and `pnpm build`.

## Deploy

`pnpm deploy` rebuilds the viewer and deploys the static assets with the Wrangler CLI available in the local development environment. Cloudflare serves the generated viewer at the Worker root and falls back to `index.html` for client-side routes.

Deployment tooling was verified with local Wrangler 4.147.0; the scripts use the Wrangler installation available on PATH.
