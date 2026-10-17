# Investigation log — `aquaflux_ui/`

> Reference-only, never auto-loads. Dated measurements, hand checks and findings behind
> `.claude/rules/ui.md`; the rule holds the current state.

## From `ui.md` — scope history

Built 2026-10-05 (v1, Results viewer + Setup case editor), Run section 2026-10-07. Renamed from
`aquaflux_viz` / `aquaflux-viz` / the `viz` extra the same day, before it was ever committed (the project
owner's call, once it stopped being only a viewer). Agreed scope (project owner, 2026-10-05): top-level
package, tests run in CI, residual history output added to the solver, hexahedra written as
`VTK_HEXAHEDRON` (see `io.md`); then a case-file editor in Setup with every option generated from the
solver's schema; then (2026-10-07) a Run section that starts a run and follows it live.

## From `ui.md` — findings behind current rules

- Automatic colour range from the focused layers: found by the project owner on the bunny room,
  2026-10-05: with the range over all layers the ceiling faces beside the lamp (G up to 45 W/m²) set the
  scale while a slice through the room held 0.003–0.06, so every slice was the bottom colour. The slice
  itself was verified correct (its values equal the cut cells' `G` exactly).
- Empty tooltips (project owner, 2026-10-06): diagnosed by reading the bubble's computed colours in the
  browser — the first guess (the tooltip's content slot) was wrong.
- Defaults display: of the 165 case-file fields whose stored default is `None` (2026-10-06), 62 resolve,
  24 are off, and 79 stay "not set" (they depend on the solve path or the rest of the case, or their
  default is an inline literal no declaration can read).
- Save as re-basing: found in the browser on the first Save as (the copy pointed at a mesh that did not
  exist).
- Nesting display: the project owner chose options 1+2 of three mocked (2026-10-06); the third, drill-in
  past depth 2 with a breadcrumb, was deferred.
- Stop grace: on pitzDaily a 3 s wait before kill was not enough for an interrupt to land.
- Mesh view checked in the browser on pitzDaily (2026-10-06): a chip highlights its patch. Through the
  persistent worker, pitzDaily's (12 225 cells) first mesh load costs 2.0 s and a repeat 0.3 s
  (2026-10-06, under a load average of ~12; see `case.md` for the per-command breakdown).
- Run section checked by hand on a scratch copy of pitzDaily (2026-10-07): live residual, tiles, status,
  the events as refits landed, the earlier-run view, the replace switch, Stop.
- Per-equation palette: on the light surface the worst adjacent pair (green/red, CVD ΔE 7.2) is legal
  only with a second encoding (hence end labels); dark passes outright (2026-10-07, the dataviz validator).

## From `ui.md` — mutation pass (2026-10-05)

All 15 targeted mutations of the package (and 12 more of the multi-slice/threshold and control logic)
(caching keys, identity reuse, multiblock naming, log floor, threshold bounds, run-record listing, the
jax-import guard, …) turn the suite red.

## From `ui.md` — Measured (2026-10-05)

Configuration: VTK 9.7.1, PyVista 0.49.0, trame 4.0.0, trame-vtk 2.11.17, macOS arm64 (11 cores,
19 GB), Python 3.13.
- Page checked by hand in the in-app browser on a 48 000-cell structured grid (run info, colour by,
  log scale, convergence plot from a real `StepHistory` file) and on the 2.46M-cell bunny room (two
  slices on different axes, a typed slice coordinate, a threshold with a typed minimum, grey context).
- See `io.md` for the hexahedron writer's file-size/memory measurement (1M cells: 70 vs 198 MB file,
  393 vs 1074 MB peak RSS).
