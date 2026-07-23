# ChefBot recipe data

Large dumps (`recipes.json`, `recipes.csv`) are **gitignored** (~80MB+).

## Quick start (reproducible demo)

The repo ships a tracked sample:

- `sample_recipes.jsonl` — 200 structured recipes (same schema as production)

```bash
python -u prepare_dataset.py --from-sample
```

This writes `recipes.json` for `ingest.py`.

## Full corpus (~62k recipes)

If you already have the dump locally, place it here as:

```text
dataset/recipes.json   # JSONL (one recipe object per line) preferred
```

Or download from a URL you host (GitHub Release, Drive, etc.):

```bash
python -u prepare_dataset.py --url "https://YOUR-HOST/recipes.jsonl" --force
# or
set CHEFBOT_DATASET_URL=https://YOUR-HOST/recipes.jsonl
python -u prepare_dataset.py --force
```

Schema (each object):

| Field | Type |
|---|---|
| `recipe_title` | string |
| `ingredients` | string[] |
| `directions` | string[] |
| `category` / `subcategory` | string (optional) |
| `description` | string (optional) |
| `num_ingredients` / `num_steps` | int (optional) |

`ingest.py` also accepts a JSON array file or `dataset/2_Recipe_json.json`.
