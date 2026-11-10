# Third-party notices

Dashboard palette/layout concepts draw on the MIT-licensed Codexa v1 review interface (Copyright 2026 Codexa v1 contributors). The new training web dashboard and event integration are independently implemented. No v2 source is incorporated.

uPlot 1.6.32 (Leon Sorokin) is bundled locally; its full MIT notice is preserved in codexa/dashboard/static/UPLOT_LICENSE. No external analytics or CDN runtime is used.

The terminal training dashboard adapts the Rich panel composition, status palette, sparklines and small rendering helpers from Codexa v1 monitoring/render.py (Copyright 2026 Codexa v1 contributors, MIT). The full original notice is retained at codexa/training/terminal/V1_LICENSE and included in package license metadata. V3 supplies its own event-state mapping and isolated renderer lifecycle; no v2 code or private assets are copied.

Original evaluation source fixtures in `codexa/evaluation/assets` are authored CC0 material, independent of acquired corpora. Their CC0 notice is retained in [the asset license](codexa/evaluation/assets/LICENSE) and included in wheel/source distributions. Code remains Apache-2.0; this fixture dedication grants no permissions for external datasets or weights.
