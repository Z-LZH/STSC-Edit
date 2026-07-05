#!/usr/bin/env python3
"""
External-mask adapter for the existing StructEdit PowerPaint pipeline.

This file does NOT modify:
    scripts/structedit/run_pipeline_Powerpaint.py

It temporarily places the supplied external mask at the path expected by the
existing pipeline:

    outputs/command/<image_id>/06_sam2_masks/combined_mask.png

Then it invokes the original pipeline with:

    --skip-selection-steps --force-edit

After the edit finishes, the original files are restored.

Typical usage:

python scripts/structedit/run_pipeline_Powerpaint_external_mask.py \
  --image-id 77 \
  --image-root data/vg/no_edit \
  --image-ext jpg \
  --cmd "remove the second computer" \
  --powerpaint-output-root outputs/test_external_mask_77 \
  --device cuda \
  --external-mask-path outputs/edit_verification_baseline_vs_vg/baseline_masks/77_second_computer/DINO_full_phrase.png
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image


def infer_operation(command: str) -> str:
    text = command.strip().lower()
    if text.startswith("add ") or " add " in f" {text} ":
        return "add"
    if text.startswith("replace ") or " replace " in f" {text} ":
        return "replace"
    if text.startswith("remove ") or text.startswith("delete "):
        return "delete"
    if text.startswith("move ") or " move " in f" {text} ":
        return "move"
    if text.startswith("change "):
        if any(token in text for token in [
            " red", " blue", " green", " yellow", " black", " white",
            " brown", " gray", " grey", " purple", " orange", " pink",
        ]):
            return "color"
        return "replace"
    return "replace"


def backup_file(path: Path, backup_dir: Path, registry: Dict[Path, Optional[Path]]) -> None:
    if path in registry:
        return

    if path.exists():
        backup_path = backup_dir / f"{len(registry):04d}_{path.name}"
        shutil.copy2(path, backup_path)
        registry[path] = backup_path
    else:
        registry[path] = None


def restore_files(registry: Dict[Path, Optional[Path]]) -> None:
    for path, backup_path in reversed(list(registry.items())):
        try:
            if backup_path is None:
                if path.exists():
                    path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_path, path)
        except Exception as exc:
            print(f"[WARN] Failed to restore {path}: {exc}", file=sys.stderr)


def normalize_external_mask(
    source_mask: Path,
    image_path: Path,
    destination_mask: Path,
) -> None:
    if not source_mask.is_file():
        raise FileNotFoundError(f"External mask not found: {source_mask}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    with Image.open(image_path) as image:
        image_size = image.size

    with Image.open(source_mask) as mask_image:
        mask = mask_image.convert("L")
        if mask.size != image_size:
            print(
                f"[ExternalMask] resize mask from {mask.size} to {image_size} "
                "with nearest-neighbor interpolation"
            )
            mask = mask.resize(image_size, resample=Image.Resampling.NEAREST)

        # Convert any non-zero mask to a clean 0/255 binary mask.
        mask = mask.point(lambda value: 255 if value > 0 else 0, mode="L")

    destination_mask.parent.mkdir(parents=True, exist_ok=True)
    mask.save(destination_mask)


def run_rule_parser(
    python_executable: str,
    rule_parser_script: Path,
    image_id: str,
    image_root: str,
    image_ext: str,
    command_text: str,
    parsed_task_path: Path,
    extra_parser_args: List[str],
) -> None:
    if not rule_parser_script.is_file():
        if parsed_task_path.is_file():
            print(
                f"[WARN] Rule parser not found: {rule_parser_script}. "
                f"Reuse existing parsed task: {parsed_task_path}"
            )
            return
        raise FileNotFoundError(
            f"Rule parser not found and parsed task does not exist:\n"
            f"  parser: {rule_parser_script}\n"
            f"  parsed task: {parsed_task_path}"
        )

    parser_command = [
        python_executable,
        str(rule_parser_script),
        "--image-id",
        image_id,
        "--image-root",
        image_root,
        "--image-ext",
        image_ext,
        "--cmd",
        command_text,
        *extra_parser_args,
    ]

    print("[ExternalMask] prepare parsed command:")
    print(" ", " ".join(parser_command))

    result = subprocess.run(parser_command, check=False)

    if result.returncode != 0:
        if parsed_task_path.is_file():
            print(
                "[WARN] Rule parser returned a non-zero status, but an existing "
                f"parsed task is available and will be reused: {parsed_task_path}"
            )
            return
        raise subprocess.CalledProcessError(result.returncode, parser_command)

    if not parsed_task_path.is_file():
        raise FileNotFoundError(
            "Rule parser finished but did not create the expected parsed task:\n"
            f"  {parsed_task_path}"
        )


def write_external_mask_meta(
    meta_path: Path,
    mask_path: Path,
    source_mask_path: Path,
    image_id: str,
    command_text: str,
) -> None:
    operation = infer_operation(command_text)
    data = {
        "external_mask": {
            "decision_key": "external_mask",
            "operation": operation,
            "mask_type": "external_binary_mask",
            "mask_path": str(mask_path),
            "source_mask_path": str(source_mask_path),
            "image_id": image_id,
            "raw_command": command_text,
        }
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use an external mask with the existing StructEdit PowerPaint "
            "pipeline without modifying run_pipeline_Powerpaint.py."
        )
    )

    parser.add_argument("--external-mask-path", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--image-root", default="data/vg/no_edit")
    parser.add_argument("--image-ext", default="jpg")
    parser.add_argument("--cmd", required=True)
    parser.add_argument("--powerpaint-output-root", required=True)
    parser.add_argument("--device", default="cuda")

    parser.add_argument(
        "--original-pipeline",
        default="scripts/structedit/run_pipeline_Powerpaint.py",
    )
    parser.add_argument(
        "--rule-parser-script",
        default="scripts/structedit/rule_parser.py",
    )
    parser.add_argument(
        "--command-output-root",
        default="outputs/command",
        help="Root used by the existing command-mode pipeline.",
    )
    parser.add_argument(
        "--skip-rule-parser",
        action="store_true",
        help="Reuse an existing 00_parsed_task.json instead of rerunning rule_parser.py.",
    )
    parser.add_argument(
        "--keep-staged-files",
        action="store_true",
        help="Do not restore the original parsed task and SAM2 mask after editing.",
    )
    parser.add_argument(
        "--rule-parser-extra-arg",
        action="append",
        default=[],
        help="Additional argument passed to rule_parser.py; may be repeated.",
    )

    # Unknown arguments are forwarded to the original pipeline.
    args, forwarded_args = parser.parse_known_args()

    original_pipeline = Path(args.original_pipeline)
    rule_parser_script = Path(args.rule_parser_script)
    external_mask_path = Path(args.external_mask_path).resolve()

    if not original_pipeline.is_file():
        raise FileNotFoundError(
            f"Original PowerPaint pipeline not found: {original_pipeline}"
        )

    image_path = (
        Path(args.image_root)
        / f"{args.image_id}.{args.image_ext.lstrip('.')}"
    )

    sample_dir = Path(args.command_output_root) / str(args.image_id)
    parsed_task_path = sample_dir / "00_parsed_task.json"
    mask_dir = sample_dir / "06_sam2_masks"
    combined_mask_path = mask_dir / "combined_mask.png"
    external_copy_path = mask_dir / "external_mask.png"
    mask_meta_path = mask_dir / "06_sam2_mask_meta.json"

    sample_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    backup_registry: Dict[Path, Optional[Path]] = {}

    with tempfile.TemporaryDirectory(prefix="structedit_external_mask_") as temp_dir:
        backup_dir = Path(temp_dir)

        # Back up every project file that this adapter may temporarily replace.
        for path in [
            parsed_task_path,
            combined_mask_path,
            external_copy_path,
            mask_meta_path,
        ]:
            backup_file(path, backup_dir, backup_registry)

        try:
            if not args.skip_rule_parser:
                run_rule_parser(
                    python_executable=sys.executable,
                    rule_parser_script=rule_parser_script,
                    image_id=str(args.image_id),
                    image_root=args.image_root,
                    image_ext=args.image_ext,
                    command_text=args.cmd,
                    parsed_task_path=parsed_task_path,
                    extra_parser_args=args.rule_parser_extra_arg,
                )
            elif not parsed_task_path.is_file():
                raise FileNotFoundError(
                    "--skip-rule-parser was specified, but parsed task is missing:\n"
                    f"  {parsed_task_path}"
                )

            normalize_external_mask(
                source_mask=external_mask_path,
                image_path=image_path,
                destination_mask=combined_mask_path,
            )
            shutil.copy2(combined_mask_path, external_copy_path)

            write_external_mask_meta(
                meta_path=mask_meta_path,
                mask_path=combined_mask_path,
                source_mask_path=external_mask_path,
                image_id=str(args.image_id),
                command_text=args.cmd,
            )

            pipeline_command = [
                sys.executable,
                str(original_pipeline),
                "--image-id",
                str(args.image_id),
                "--image-root",
                args.image_root,
                "--image-ext",
                args.image_ext,
                "--cmd",
                args.cmd,
                "--skip-selection-steps",
                "--force-edit",
                "--powerpaint-output-root",
                args.powerpaint_output_root,
                "--device",
                args.device,
                *forwarded_args,
            ]

            print("=" * 100)
            print("[ExternalMask] source mask:", external_mask_path)
            print("[ExternalMask] staged mask:", combined_mask_path)
            print("[ExternalMask] parsed task:", parsed_task_path)
            print("[ExternalMask] output root:", args.powerpaint_output_root)
            print("[ExternalMask] invoke original pipeline:")
            print(" ", " ".join(pipeline_command))
            print("=" * 100)

            subprocess.run(pipeline_command, check=True)

        finally:
            if args.keep_staged_files:
                print("[ExternalMask] keep staged files under:", sample_dir)
            else:
                restore_files(backup_registry)
                print("[ExternalMask] restored original command files.")

    print("[ExternalMask] edit finished successfully.")


if __name__ == "__main__":
    main()
