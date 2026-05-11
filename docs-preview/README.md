# docs-preview

Minimal Cloudflare Worker that hosts a [pydantic/unified-docs](https://github.com/pydantic/unified-docs) build as static assets.

Used by `.github/workflows/docs-preview.yml` to deploy per-PR docs previews when a maintainer adds the `trigger:docs` label. Each preview is uploaded as a [Worker version](https://developers.cloudflare.com/workers/configuration/previews/) with a stable alias derived from the PR number and commit SHA — Cloudflare auto-prunes aliases beyond the 1000 most recent, so there is no manual cleanup.

The worker has no `main` script — it only serves static assets under `../../unified-docs/dist` (set by the workflow, which checks out unified-docs as a sibling of the pydantic-ai PR checkout).

## First-time setup

`wrangler versions upload` (used by the preview workflow) fails on first-ever deploy of a Worker name. Before the first PR preview, dispatch the `Docs Preview Bootstrap` workflow once from the Actions tab — it does a single `wrangler deploy` of a placeholder to create the `pai-docs-preview` Worker on Cloudflare. Re-run only if the Worker is ever deleted.

## Production docs

Production docs at <https://ai.pydantic.dev/> are deployed via `pydantic/unified-docs` itself (triggered from `manually-deploy-docs.yml` and the tag-time `deploy-docs` job in `ci.yml`).
