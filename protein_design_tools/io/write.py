# protein_design_tools/io/write.py
from __future__ import annotations

from pathlib import Path
from typing import Union

from ..core.protein_structure import ProteinStructure


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_PDB_ATOM_FMT = (
    "ATOM  {atom_id:5d} {name:^4}{alt_loc:1}{res_name:>3} {chain_id:1}"
    "{res_seq:4d}{i_code:1}   "
    "{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}"
    "          {element:>2}{charge:2}\n"
)


def _write_pdb(struct: ProteinStructure, fh) -> None:
    """Write *struct* as a PDB file (very thin, ATOM/HETATM only)."""
    for ch in struct.chains:  # type: Chain
        for res in ch.residues:  # type: Residue
            for at in res.atoms:  # type: Atom
                fh.write(
                    _PDB_ATOM_FMT.format(
                        atom_id=at.atom_id or 0,
                        name=at.name[:4],
                        alt_loc=at.alt_loc[:1] or " ",
                        res_name=res.name[:3],
                        chain_id=ch.name[:1] or " ",
                        res_seq=res.res_seq,
                        i_code=res.i_code[:1] or " ",
                        x=at.x,
                        y=at.y,
                        z=at.z,
                        occupancy=at.occupancy or 0.0,
                        temp_factor=at.temp_factor or 0.0,
                        element=(at.element or at.name[0]).rjust(2),
                        charge=at.charge.rjust(2) if at.charge else "  ",
                    )
                )
    fh.write("END\n")


def _write_cif(struct: ProteinStructure, fh) -> None:
    """Very minimal mmCIF writer with a single loop_ for atom_site."""
    # headers
    fh.write("data_generated_by_protein_design_tools\n")
    fh.write("loop_\n")
    cols = (
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
    )
    for c in cols:
        fh.write(f"{c}\n")

    idx = 1
    for ch in struct.chains:
        for res in ch.residues:
            for at in res.atoms:
                fh.write(
                    f"ATOM {idx:d} {at.element or at.name[0]} {at.name} "
                    f"{res.name} {ch.name} {res.res_seq:d} "
                    f"{at.x:.3f} {at.y:.3f} {at.z:.3f}\n"
                )
                idx += 1


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def write_structure(
    structure: ProteinStructure,
    file_path: Union[str, Path],
    *,
    include_hydrogens: bool = False,
    include_hetatm: bool = True,
) -> None:
    """
    Dump *structure* to PDB.  By default hydrogens are dropped and only ATOM
    records are written.
    """

    def _format_atom(rec, serial):
        altloc = " " if rec.alt_loc in (".", "?", "") else rec.alt_loc
        icode = " " if rec.i_code in (".", "?", "") else rec.i_code
        return (
            f"{rec.record:<6}{serial:5d} "
            f"{rec.name:^4}{altloc}"
            f"{rec.residue.name:>3} {rec.chain_id:1}"
            f"{rec.residue.res_seq:>4}{icode}   "
            f"{rec.x:8.3f}{rec.y:8.3f}{rec.z:8.3f}"
            f"{rec.occupancy:6.2f}{rec.temp_factor:6.2f}          "
            f"{rec.element:>2}\n"
        )

    serial = 1
    lines = []
    for ch in structure.chains:
        for res in ch.residues:
            is_het = res.name not in ProteinStructure.STANDARD_RESIDUES
            if is_het and not include_hetatm:
                continue
            for at in res.atoms:
                if at.element.upper() == "H" and not include_hydrogens:
                    continue
                lines.append(_format_atom(at, serial))
                serial += 1

    Path(file_path).write_text("".join(lines))
