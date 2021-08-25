import io
import pdbfixer
from simtk.openmm import app
from simtk.openmm.app import element
import time
import simtk.openmm as openmm
import numpy as np
from simtk import unit

ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms

def fix_pdb(pdbfile, alterations_info):
    """Apply pdbfixer to the contents of a PDB file; return a PDB string result.

    1) Replaces nonstandard residues.
    2) Removes heterogens (non protein residues) including water.
    3) Adds missing residues and missing atoms within existing residues.
    4) Adds hydrogens assuming pH=7.0.
    5) KeepIds is currently true, so the fixer must keep the existing chain and
       residue identifiers. This will fail for some files in wider PDB that have
       invalid IDs.

    Args:
      pdbfile: Input PDB file handle.
      alterations_info: A dict that will store details of changes made.

    Returns:
      A PDB string representing the fixed structure.
    """

    fixer = pdbfixer.PDBFixer(pdbfile=pdbfile)

    fixer.findNonstandardResidues()

    alterations_info['nonstandard_residues'] = fixer.nonstandardResidues
    fixer.replaceNonstandardResidues()
    _remove_heterogens(fixer, alterations_info, keep_water=False)

    fixer.findMissingResidues()
    alterations_info['missing_residues'] = fixer.missingResidues
    fixer.findMissingAtoms()
    alterations_info['missing_heavy_atoms'] = fixer.missingAtoms
    alterations_info['missing_terminals'] = fixer.missingTerminals

    ################################# time comsuption
    fixer.addMissingAtoms(seed=0)
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(platform=openmm.Platform.getPlatformByName("CUDA"))
    fixer.topology = modeller.topology
    fixer.positions = modeller.positions

    #################################
    # print(time.time() - begin)

    out_handle = io.StringIO()
    app.PDBFile.writeFile(fixer.topology, fixer.positions, out_handle,
                          keepIds=True)
    return out_handle.getvalue()


def clean_structure(pdb_structure, alterations_info):
    """Applies additional fixes to an OpenMM structure, to handle edge cases.

    Args:
      pdb_structure: An OpenMM structure to modify and fix.
      alterations_info: A dict that will store details of changes made.
    """
    _replace_met_se(pdb_structure, alterations_info)
    _remove_chains_of_length_one(pdb_structure, alterations_info)


def _remove_heterogens(fixer, alterations_info, keep_water):
    """Removes the residues that Pdbfixer considers to be heterogens.

    Args:
      fixer: A Pdbfixer instance.
      alterations_info: A dict that will store details of changes made.
      keep_water: If True, water (HOH) is not considered to be a heterogen.
    """
    initial_resnames = set()
    for chain in fixer.topology.chains():
        for residue in chain.residues():
            initial_resnames.add(residue.name)
    fixer.removeHeterogens(keepWater=keep_water)
    final_resnames = set()
    for chain in fixer.topology.chains():
        for residue in chain.residues():
            final_resnames.add(residue.name)
    alterations_info['removed_heterogens'] = (
        initial_resnames.difference(final_resnames))


def _replace_met_se(pdb_structure, alterations_info):
    """Replace the Se in any MET residues that were not marked as modified."""
    modified_met_residues = []
    for res in pdb_structure.iter_residues():
        name = res.get_name_with_spaces().strip()
        if name == 'MET':
            s_atom = res.get_atom('SD')
            if s_atom.element_symbol == 'Se':
                s_atom.element_symbol = 'S'
                s_atom.element = element.get_by_symbol('S')
                modified_met_residues.append(s_atom.residue_number)
    alterations_info['Se_in_MET'] = modified_met_residues


def _remove_chains_of_length_one(pdb_structure, alterations_info):
    """Removes chains that correspond to a single amino acid.

    A single amino acid in a chain is both N and C terminus. There is no force
    template for this case.

    Args:
      pdb_structure: An OpenMM pdb_structure to modify and fix.
      alterations_info: A dict that will store details of changes made.
    """
    removed_chains = {}
    for model in pdb_structure.iter_models():
        valid_chains = [c for c in model.iter_chains() if len(c) > 1]
        invalid_chain_ids = [c.chain_id for c in model.iter_chains() if len(c) <= 1]
        model.chains = valid_chains
        for chain_id in invalid_chain_ids:
            model.chains_by_id.pop(chain_id)
        removed_chains[model.number] = invalid_chain_ids
    alterations_info['removed_chains'] = removed_chains

def _check_cleaned_atoms(pdb_cleaned_string: str, pdb_ref_string: str):
    """Checks that no atom positions have been altered by cleaning."""
    cleaned = app.PDBFile(io.StringIO(pdb_cleaned_string))
    reference = app.PDBFile(io.StringIO(pdb_ref_string))

    cl_xyz = np.array(cleaned.getPositions().value_in_unit(LENGTH))
    ref_xyz = np.array(reference.getPositions().value_in_unit(LENGTH))

    for ref_res, cl_res in zip(reference.topology.residues(),
                               cleaned.topology.residues()):
        assert ref_res.name == cl_res.name
        for rat in ref_res.atoms():
            for cat in cl_res.atoms():
                if cat.name == rat.name:
                    if not np.array_equal(cl_xyz[cat.index], ref_xyz[rat.index]):
                        raise ValueError(f"Coordinates of cleaned atom {cat} do not match "
                                         f"coordinates of reference atom {rat}.")