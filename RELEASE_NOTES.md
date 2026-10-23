## What's new in v0.3.4

### 📦 Export
- **Excel copy of the metadata** — new "Also export Excel (.xlsx)" switch in the Export dialog adds `{acronym}_metadata.xlsx` to the ZIP, next to the CSV. Excel opens it directly, with no separator or encoding surprises
- **Safer metadata CSV** — every field is quoted, line breaks inside a value (e.g. text read by the AI) become spaces, and the file is UTF-8 with BOM, so commas, quotes and accents no longer break the table when the CSV is opened in Excel or imported elsewhere
- `openpyxl` added to the requirements

## What's new in v0.3.0

### ✏️ Annotation tab
- **Nested vessel support** — new Polygon tool (📐) to manually outline vessels drawn inside other vessels. Polygons are saved as a sidecar JSON and extracted as independent cards; the inner vessel area is automatically subtracted (whitened) from the containing card
- **Zoom & pan** — canvas fits the full page on load; zoom in/out with +/−/fit buttons or the mouse wheel
- **Brush/eraser cursor ring** — a visible size indicator follows the cursor so you always know the affected area
- **Colorize mode (🎨)** — toggle a political-map coloring where every disconnected mask region gets a distinct hue; fused/merged regions share a color, making accidents immediately visible

### 📊 Tabular tab
- **AI extraction improvements** — dynamic token budget prevents JSON truncation on dense pages; robust parser recovers partial responses
- **Canonical column names** — write column names in `[square brackets]` in the prompt to lock the exact JSON key the model will use, preventing key drift across batch runs
- **Numbers from crops** — optional toggle to extract per-drawing values (e.g. inventory numbers) by cropping and upscaling each individual figure; applies to both single-page and batch extraction
- **Box visibility toggle** — iOS-style switch to show/hide bounding boxes on the canvas
- **Clear Table** — reset all values on the current page without losing the column structure
- **Export Combined CSV removed** — tabular data is now read directly from `cards/mask_info.csv` at export time, no manual intermediate step needed

### 🔍 Post Processing tab
- **Grid view** — all extracted pieces displayed together at real relative sizes, replacing the old one-at-a-time navigator
- **Per-card hover controls** — flip vertical, flip horizontal, ENT/FRAG toggle, exclude from export — all without leaving the grid
- **Border color coding** — blue = ENT, orange = FRAG for at-a-glance classification review
- **Lightbox** — click any card to open a full-screen zoom view
- **Exclude from export** — cards marked with ✕ are skipped in the ZIP (images and metadata rows)
- **Progress bar** — only visible while "Process All Images" is running
- **Auto-flip toggles** — replaced plain checkboxes with iOS-style switches

### 📦 Export
- All metadata fields written as plain text in the output CSV (no implicit numeric coercion)

### 🐛 Bug fixes
- Fixed flip endpoint applying to the wrong card when filenames sort lexicographically vs. naturally
- Fixed small vessel area filter (was 4× too large due to counting RGBA channels); lowered minimum area ratio to 0.0002
