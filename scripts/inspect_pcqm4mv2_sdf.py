from __future__ import annotations

import argparse

from mkd_mpp.pcqm4mv2 import iter_sdf_molecules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect PCQM4Mv2 SDF molecules without preprocessing.")
    parser.add_argument("--sdf-path", required=True)
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for idx, mol in iter_sdf_molecules(args.sdf_path):
        if idx >= args.limit:
            break
        print(f"{idx}-th rdkit mol obj: {mol}")


if __name__ == "__main__":
    main()
