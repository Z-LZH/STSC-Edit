# StructEdit

StructEdit is an image editing project that uses text instructions to locate target objects in images and perform editing operations such as replacement, movement, color change, object addition, and object removal.

This repository contains the core scripts, several example images, and two example task files.

## Installation

Python 3.10 is recommended.

```bash
conda create -n structedit python=3.10 -y
conda activate structedit
pip install -r requirements.txt
```

Some pipelines may also require additional dependencies, such as:

- GroundingDINO
- SAM2
- PowerPaint
- FLUX / SDXL related dependencies

## Example Images

Example images are placed in:

```text
examples/images/
```

The example images are:

```text
examples/images/1.jpg
examples/images/2.jpg
examples/images/3.jpg
examples/images/4.jpg
examples/images/5.jpg
```

## Task Files

Task files are placed in:

```text
examples/tasks/
```

This repository keeps two task files:

```text
examples/tasks/demo_direct_edit_tasks.jsonl
examples/tasks/demo_selection_tasks.jsonl
```

### demo_direct_edit_tasks.jsonl

This file is used for direct image editing tasks.

```jsonl
{"id":"demo_01_top_knife","image_id":"1","edit_cmd":"replace the top knife with a spoon","tag":"replace_top_knife_with_spoon"}
{"id":"demo_02_left_potato","image_id":"2","edit_cmd":"move the leftmost potato on the plate to the right side of the plate","tag":"move_left_potato_to_right_side"}
{"id":"demo_03_left_chair","image_id":"3","edit_cmd":"change the color of the left chair to red","tag":"change_left_chair_to_red"}
{"id":"demo_04_left_flower","image_id":"4","edit_cmd":"add a butterfly on the flower on the left","tag":"add_butterfly_on_left_flower"}
{"id":"demo_05_leftmost_elephant","image_id":"5","edit_cmd":"remove the leftmost elephant","tag":"remove_leftmost_elephant"}
```

### demo_selection_tasks.jsonl

This file is used for target selection and object localization tasks.

```jsonl
{"id":"demo_01_top_knife","image":"examples/images/1.jpg","target_phrase":"the top of knife","category":"knife"}
{"id":"demo_02_left_potato","image":"examples/images/2.jpg","target_phrase":"the leftmost potato on the plate","category":"potato"}
{"id":"demo_03_left_chair","image":"examples/images/3.jpg","target_phrase":"the left chair","category":"chair"}
{"id":"demo_04_left_flower","image":"examples/images/4.jpg","target_phrase":"the flower on the left","category":"flower"}
{"id":"demo_05_leftmost_elephant","image":"examples/images/5.jpg","target_phrase":"the leftmost elephant","category":"elephant"}
```

## Field Description

Fields in `demo_direct_edit_tasks.jsonl`:

- `id`: task name
- `image_id`: image index, corresponding to the image file in `examples/images/`
- `edit_cmd`: text editing instruction
- `tag`: short label used for output folders or logs

Fields in `demo_selection_tasks.jsonl`:

- `id`: task name
- `image`: image path
- `target_phrase`: target object description
- `category`: target object category

## Quick Test

Check Python scripts:

```bash
python -m compileall -q scripts
```

Check shell scripts:

```bash
bash -n scripts/run_edit_verification.sh \
  scripts/run_position_ablation.sh \
  scripts/run_selection_baselines_all.sh \
  scripts/structedit/run_object_selection_demo.sh
```

## Run Examples

### Run target localization

```bash
python scripts/structedit/run_structloc_pipeline.py \
  --image examples/images/1.jpg \
  --cmd "replace the top of knife with a spoon" \
  --sample-id demo_01_top_knife \
  --out-dir outputs/demo_01_top_knife \
  --device cuda \
  --skip-flux
```

### Run PowerPaint editing

```bash
python scripts/structedit/run_pipeline_Powerpaint.py \
  --image-id 1 \
  --image-root examples/images \
  --image-ext jpg \
  --cmd "replace the top of knife with a spoon" \
  --powerpaint-output-root outputs/powerpaint_demo_01 \
  --device cuda
```

## Outputs

The output results are saved under:

```text
outputs/
```

Example output folders:

```text
outputs/demo_01_top_knife/
outputs/powerpaint_demo_01/
```

## Notes

- The example images are provided for quick testing.
- Full image editing usually requires a GPU environment.
- If model files are missing, prepare the required checkpoints first.
- If packages such as `groundingdino`, `sam2`, or `PowerPaint` are missing, install the corresponding projects first.

## License

This project is released under the MIT License.
