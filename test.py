#!/usr/bin/env python3
"""
Shrinks and recolors all chess piece SDF models so they fit the board squares
and match the chessboard's colour scheme.

Usage:
    python3 fix_chess_pieces.py --models-dir /path/to/worlds/models
    python3 fix_chess_pieces.py --models-dir /path/to/worlds/models --scale-factor 0.4

A .bak copy of each model.sdf is made before editing (skipped if a .bak
already exists, so re-running won't overwrite your original backup).
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# Multiplies every existing <scale> and the matching collision box/pose.
# Current mesh scale is 0.03 0.03 0.03 -> 0.7 shrinks it to 0.021 (70% of original). Tweak and re-run until it fits.
DEFAULT_SCALE_FACTOR = 0.9

# Edit these to match your actual chessboard square colours (RGBA, 0-1 range).
WHITE_PIECE_COLOR = {
    "ambient":  "0.85 0.80 0.65 1",
    "diffuse":  "0.90 0.85 0.70 1",
    "specular": "0.9 0.9 0.9 1",
}
BLACK_PIECE_COLOR = {
    "ambient":  "0.25 0.15 0.08 1",
    "diffuse":  "0.35 0.22 0.12 1",
    "specular": "0.4 0.4 0.4 1",
}

PIECE_TYPES = ["pawn", "rook", "knight", "bishop", "queen", "king"]
PREFIX_COLOR = {"w": WHITE_PIECE_COLOR, "b": BLACK_PIECE_COLOR}


def scale_triplet(text: str, factor: float) -> str:
    parts = [float(p) * factor for p in text.split()]
    return " ".join(f"{p:.6f}" for p in parts)


def set_color(material_elem: ET.Element, color: dict):
    for tag in ("ambient", "diffuse", "specular"):
        el = material_elem.find(tag)
        if el is not None:
            el.text = color[tag]


def fix_model_sdf(sdf_path: Path, color: dict, scale_factor: float) -> bool:
    backup_path = sdf_path.with_suffix(".sdf.bak")
    if not backup_path.exists():
        shutil.copy2(sdf_path, backup_path)

    tree = ET.parse(sdf_path)
    root = tree.getroot()
    changed = False

    # Shrink the mesh
    for scale_el in root.iter("scale"):
        scale_el.text = scale_triplet(scale_el.text, scale_factor)
        changed = True

    # Shrink the collision box + its pose so it still wraps the smaller mesh
    for collision in root.iter("collision"):
        pose_el = collision.find("pose")
        if pose_el is not None:
            values = pose_el.text.split()
            xyz = [float(v) * scale_factor for v in values[:3]]
            rpy = values[3:]  # rotation untouched
            pose_el.text = " ".join(f"{v:.6f}" for v in xyz) + " " + " ".join(rpy)
            changed = True
        size_el = collision.find("geometry/box/size")
        if size_el is not None:
            size_el.text = scale_triplet(size_el.text, scale_factor)
            changed = True

    # Recolor to match the board
    for material in root.iter("material"):
        set_color(material, color)
        changed = True

    if changed:
        try:
            ET.indent(tree, space="  ")  # Python 3.9+, pretty-prints the output
        except AttributeError:
            pass
        tree.write(sdf_path, encoding="UTF-8", xml_declaration=True)
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", required=True, help="Path to worlds/models")
    parser.add_argument("--scale-factor", type=float, default=DEFAULT_SCALE_FACTOR)
    args = parser.parse_args()

    models_dir = Path(args.models_dir)

    for prefix, color in PREFIX_COLOR.items():
        for piece in PIECE_TYPES:
            sdf_path = models_dir / f"{prefix}_{piece}" / "model.sdf"
            if not sdf_path.exists():
                print(f"skip (not found): {sdf_path}")
                continue
            changed = fix_model_sdf(sdf_path, color, args.scale_factor)
            print(f"{'updated' if changed else 'no change'}: {sdf_path}")


if __name__ == "__main__":
    main()