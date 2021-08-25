import io
import time
from typing import Collection, Optional, Sequence
import numpy as np
from simtk import openmm
from simtk import unit
from simtk.openmm import app as app
# from simtk.openmm.app.internal.pdbstructure import PdbStructure
# from . import cleanup
from ..common.protein import Protein
from . import utils


from ..utils import cost

from ..common import residue_constants
import torch
from torch.nn.functional import pad

ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms

force_field = app.ForceField("amber99sb.xml")
constraints = app.HBonds

try:
    platform=openmm.Platform.getPlatformByName("CUDA")
except:
    try:
        platform = openmm.Platform.getPlatformByName("OpenCL")
    except:
        platform=openmm.Platform.getPlatformByName("CPU")

def get_violation_metrics(prot_np,
                          tolerance_factor=12.0,
                          overlap_tolerance=1.5):
    # TODO: update function make_atom14_positions(batch)
    batch = {
        "aatype": prot_np.aatype,
        "all_atom_positions": prot_np.atom_positions.astype(np.float32),
        "all_atom_mask": prot_np.atom_mask.astype(np.float32),
        "residue_index": prot_np.residue_index,
    }
    batch["seq_mask"] = np.ones_like(batch["aatype"], np.float32)
    batch = make_atom14_positions(batch)

    # radius map
    atomtype_radius = torch.tensor(
        [residue_constants.van_der_waals_radius[name[0]] for name in residue_constants.atom_types])
    atom14_atom_radius = atomtype_radius[torch.tensor(batch['residx_atom14_to_atom37']).long().reshape(-1)].reshape(-1,
                                                                                                                    14)

    atom14_pred_positions = torch.from_numpy(batch["atom14_gt_positions"])
    aatype = torch.from_numpy(batch["aatype"])
    pred_atom_mask = torch.from_numpy(batch["atom14_atom_exists"]).bool()
    residue_index = torch.from_numpy(batch["residue_index"])

    # bond vec
    a_c = atom14_pred_positions[:-1, 1] - atom14_pred_positions[:-1, 2]
    c_n = atom14_pred_positions[:-1, 2] - atom14_pred_positions[1:, 0]
    n_a = atom14_pred_positions[1:, 0] - atom14_pred_positions[1:, 1]
    # C-N bond violations
    std = torch.where(aatype[1:] == residue_constants.resname_to_idx['PRO'],
                      residue_constants.between_res_bond_length_c_n[1],
                      residue_constants.between_res_bond_length_c_n[0])
    dev = torch.where(aatype[1:] == residue_constants.resname_to_idx['PRO'],
                      residue_constants.between_res_bond_length_stddev_c_n[1],
                      residue_constants.between_res_bond_length_stddev_c_n[0])

    mask = pred_atom_mask[:-1, 2] * pred_atom_mask[1:, 0] * ((residue_index[1:] - residue_index[:-1]) == 1.0)
    c_n_mat = ((c_n.norm(dim=-1) - std).abs() - tolerance_factor * dev).relu() * mask
    # A-C-N C-N-A angle violations
    std1 = residue_constants.between_res_cos_angles_ca_c_n[0]
    dev1 = residue_constants.between_res_bond_length_stddev_c_n[0]
    std2 = residue_constants.between_res_cos_angles_c_n_ca[0]
    dev2 = residue_constants.between_res_cos_angles_c_n_ca[1]
    a1 = torch.cosine_similarity(a_c, -c_n)
    a_c_n_mat = ((a1 - std1).abs() - tolerance_factor * dev1).relu()
    a2 = torch.cosine_similarity(c_n, -n_a)
    c_n_a_mat = ((a2 - std2).abs() - tolerance_factor * dev2).relu()

    # clash
    a = atom14_pred_positions[pred_atom_mask]
    #
    dist = (a[None] - a[:, None]).square().sum(dim=-1).sqrt()
    #
    c_n_mask = torch.zeros([14, 14])
    c_n_mask[2, 0] = 1
    c_n_mask = c_n_mask[None, :, None, ].bool()
    #
    res_mask = torch.eye(atom14_pred_positions.shape[0], dtype=torch.bool)[:, None, :, None]
    res_mask = res_mask.expand([atom14_pred_positions.shape[0], 14, atom14_pred_positions.shape[0], 14])[
                   pred_atom_mask][:, pred_atom_mask]
    #
    neigh_mask = pad(torch.eye(atom14_pred_positions.shape[0] - 1), [1, 0, 0, 1])
    neigh_mask = neigh_mask[:, None, :, None].bool()
    #
    cond_mask = (c_n_mask * neigh_mask)[pred_atom_mask][:, pred_atom_mask]
    cond_mask = cond_mask | cond_mask.T
    #
    s_mask = torch.zeros(14, 14)
    s_mask[residue_constants.restype_name_to_atom14_names['CYS'].index('SG'),
           residue_constants.restype_name_to_atom14_names['CYS'].index('SG')] = 1
    s_mask = s_mask[None, :, None].bool()
    s_s_mask = s_mask.expand([atom14_pred_positions.shape[0], 14, atom14_pred_positions.shape[0], 14])[pred_atom_mask][
               :, pred_atom_mask]
    #
    rr = (atom14_atom_radius[:, None, :, None] + atom14_atom_radius[None, :, None, :]).permute([0, 2, 1, 3])[
             pred_atom_mask][:, pred_atom_mask]
    mask = ((~(cond_mask | s_s_mask | res_mask))).triu()
    clash_mat = ((rr - overlap_tolerance - dist) * mask).relu()
    clashs = (c_n_mat > 0).sum() + (a_c_n_mat > 0).sum() + (c_n_a_mat > 0).sum() + (clash_mat > 0).sum()
    return (c_n_mat, a_c_n_mat, c_n_a_mat, clash_mat), clashs.item()







def will_restrain(atom: app.Atom, rset: str) -> bool:
    """Returns True if the atom will be restrained by the given restraint set."""

    if rset == "non_hydrogen":
        return atom.element.name != "hydrogen"
    elif rset == "c_alpha":
        return atom.name == "CA"


def _add_restraints(
        system: openmm.System,
        reference_pdb: app.PDBFile,
        stiffness: unit.Unit,
        rset: str,
        exclude_residues: Sequence[int]):
    """Adds a harmonic potential that restrains the system to a structure."""
    assert rset in ["non_hydrogen", "c_alpha"]

    force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", stiffness)
    for p in ["x0", "y0", "z0"]:
        force.addPerParticleParameter(p)

    for i, atom in enumerate(reference_pdb.topology.atoms()):
        if atom.residue.index in exclude_residues:
            continue
        if will_restrain(atom, rset):
            force.addParticle(i, reference_pdb.positions[i])
    # print("Restraining %d / %d particles.",
    #              force.getNumParticles(), system.getNumParticles())
    system.addForce(force)

@cost
def _openmm_minimize(
        pdb_str: str,
        max_iterations: int,
        tolerance: unit.Unit,
        stiffness: unit.Unit,
        restraint_set: str,
        exclude_residues: Sequence[int]):
    """Minimize energy via openmm."""

    pdb = app.PDBFile(io.StringIO(pdb_str))

    system = force_field.createSystem(
        pdb.topology, constraints=constraints)

    if stiffness > 0 * ENERGY / (LENGTH ** 2):
        _add_restraints(system, pdb, stiffness, restraint_set, exclude_residues)
    simulation = app.Simulation(
        pdb.topology, system, openmm.VerletIntegrator(0.), platform)
    simulation.context.setPositions(pdb.positions)

    ret = {}
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["einit"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["posinit"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

    simulation.minimizeEnergy(maxIterations=max_iterations,
                              tolerance=tolerance)
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["efinal"] = state.getPotentialEnergy().value_in_unit(ENERGY)

    ret["pos"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
    with io.StringIO() as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)
        ret["min_pdb"] = f.getvalue()
    return ret




def _check_residues_are_well_defined(prot):
    """Checks that all residues contain non-empty atom sets."""
    if (prot.atom_mask.sum(axis=-1) == 0).any():
        raise ValueError("Amber minimization can only be performed on proteins with"
                         " well-defined residues. This protein contains at least"
                         " one residue with no atoms.")


def make_atom14_positions(prot):
    """Constructs denser atom positions (14 dimensions instead of 37)."""
    restype_atom14_to_atom37 = []  # mapping (restype, atom14) --> atom37
    restype_atom37_to_atom14 = []  # mapping (restype, atom37) --> atom14
    restype_atom14_mask = []

    for rt in residue_constants.restypes:
        atom_names = residue_constants.restype_name_to_atom14_names[
            residue_constants.restype_1to3[rt]]

        restype_atom14_to_atom37.append([
            (residue_constants.atom_order[name] if name else 0)
            for name in atom_names
        ])

        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append([
            (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
            for name in residue_constants.atom_types
        ])

        restype_atom14_mask.append([(1. if name else 0.) for name in atom_names])

    # Add dummy mapping for restype 'UNK'.
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.] * 14)

    restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, dtype=np.int32)
    restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14, dtype=np.int32)
    restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.float32)

    # Create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein.
    residx_atom14_to_atom37 = restype_atom14_to_atom37[prot["aatype"]]
    residx_atom14_mask = restype_atom14_mask[prot["aatype"]]

    # Create a mask for known ground truth positions.
    residx_atom14_gt_mask = residx_atom14_mask * np.take_along_axis(
        prot["all_atom_mask"], residx_atom14_to_atom37, axis=1).astype(np.float32)

    # Gather the ground truth positions.
    residx_atom14_gt_positions = residx_atom14_gt_mask[:, :, None] * (
        np.take_along_axis(prot["all_atom_positions"],
                           residx_atom14_to_atom37[..., None],
                           axis=1))

    prot["atom14_atom_exists"] = residx_atom14_mask
    prot["atom14_gt_exists"] = residx_atom14_gt_mask
    prot["atom14_gt_positions"] = residx_atom14_gt_positions

    prot["residx_atom14_to_atom37"] = residx_atom14_to_atom37

    # Create the gather indices for mapping back.
    residx_atom37_to_atom14 = restype_atom37_to_atom14[prot["aatype"]]
    prot["residx_atom37_to_atom14"] = residx_atom37_to_atom14

    # Create the corresponding mask.
    restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
    for restype, restype_letter in enumerate(residue_constants.restypes):
        restype_name = residue_constants.restype_1to3[restype_letter]
        atom_names = residue_constants.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = residue_constants.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[prot["aatype"]]
    prot["atom37_atom_exists"] = residx_atom37_mask

    # As the atom naming is ambiguous for 7 of the 20 amino acids, provide
    # alternative ground truth coordinates where the naming is swapped
    restype_3 = [
        residue_constants.restype_1to3[res] for res in residue_constants.restypes
    ]
    restype_3 += ["UNK"]

    # Matrices for renaming ambiguous atoms.
    all_matrices = {res: np.eye(14, dtype=np.float32) for res in restype_3}
    for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
        correspondences = np.arange(14)
        for source_atom_swap, target_atom_swap in swap.items():
            source_index = residue_constants.restype_name_to_atom14_names[
                resname].index(source_atom_swap)
            target_index = residue_constants.restype_name_to_atom14_names[
                resname].index(target_atom_swap)
            correspondences[source_index] = target_index
            correspondences[target_index] = source_index
            renaming_matrix = np.zeros((14, 14), dtype=np.float32)
            for index, correspondence in enumerate(correspondences):
                renaming_matrix[index, correspondence] = 1.
        all_matrices[resname] = renaming_matrix.astype(np.float32)
    renaming_matrices = np.stack([all_matrices[restype] for restype in restype_3])

    # Pick the transformation matrices for the given residue sequence
    # shape (num_res, 14, 14).
    renaming_transform = renaming_matrices[prot["aatype"]]

    # Apply it to the ground truth positions. shape (num_res, 14, 3).
    alternative_gt_positions = np.einsum("rac,rab->rbc",
                                         residx_atom14_gt_positions,
                                         renaming_transform)
    prot["atom14_alt_gt_positions"] = alternative_gt_positions

    # Create the mask for the alternative ground truth (differs from the
    # ground truth mask, if only one of the atoms in an ambiguous pair has a
    # ground truth position).
    alternative_gt_mask = np.einsum("ra,rab->rb",
                                    residx_atom14_gt_mask,
                                    renaming_transform)

    prot["atom14_alt_gt_exists"] = alternative_gt_mask

    # Create an ambiguous atoms mask.  shape: (21, 14).
    restype_atom14_is_ambiguous = np.zeros((21, 14), dtype=np.float32)
    for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
        for atom_name1, atom_name2 in swap.items():
            restype = residue_constants.restype_order[
                residue_constants.restype_3to1[resname]]
            atom_idx1 = residue_constants.restype_name_to_atom14_names[resname].index(
                atom_name1)
            atom_idx2 = residue_constants.restype_name_to_atom14_names[resname].index(
                atom_name2)
            restype_atom14_is_ambiguous[restype, atom_idx1] = 1
            restype_atom14_is_ambiguous[restype, atom_idx2] = 1

    # From this create an ambiguous_mask for the given sequence.
    prot["atom14_atom_is_ambiguous"] = (
        restype_atom14_is_ambiguous[prot["aatype"]])
    return prot



@cost
def _run_one_iteration(
        *,
        pdb_string: str,
        max_iterations: int,
        tolerance: float,
        stiffness: float,
        restraint_set: str,
        max_attempts: int,
        exclude_residues: Optional[Collection[int]] = None):
    """Runs the minimization pipeline.

    Args:
      pdb_string: A pdb string.
      max_iterations: An `int` specifying the maximum number of L-BFGS iterations.
      A value of 0 specifies no limit.
      tolerance: kcal/mol, the energy tolerance of L-BFGS.
      stiffness: kcal/mol A**2, spring constant of heavy atom restraining
        potential.
      restraint_set: The set of atoms to restrain.
      max_attempts: The maximum number of minimization attempts.
      exclude_residues: An optional list of zero-indexed residues to exclude from
          restraints.

    Returns:
      A `dict` of minimization info.
    """
    exclude_residues = exclude_residues or []

    # Assign physical dimensions.
    tolerance = tolerance * ENERGY
    stiffness = stiffness * ENERGY / (LENGTH ** 2)

    start = time.time()
    minimized = False
    attempts = 0
    while not minimized and attempts < max_attempts:
        # print("attempts",attempts)
        attempts += 1
        try:
            # print("Minimizing protein, attempt %d of %d.",
            #              attempts, max_attempts)
            ret = _openmm_minimize(
                pdb_string, max_iterations=max_iterations,
                tolerance=tolerance, stiffness=stiffness,
                restraint_set=restraint_set,
                exclude_residues=exclude_residues)
            minimized = True
        except Exception as e:  # pylint: disable=broad-except
            print(e)
    if not minimized:
        raise ValueError(f"Minimization failed after {max_attempts} attempts.")
    ret["opt_time"] = time.time() - start
    ret["min_attempts"] = attempts
    return ret

@cost
def run_pipeline(
        pdb_string,
        stiffness: float=10,
        max_outer_iterations: int = 1,
        place_hydrogens_every_iteration: bool = False,
        max_iterations: int = 0,
        tolerance: float = 2.39,
        restraint_set: str = "non_hydrogen",
        max_attempts: int = 100,
        checks: bool = True,
        exclude_residues: Optional[Sequence[int]] = None):
    """Run iterative amber relax.

    Successive relax iterations are performed until all violations have been
    resolved. Each iteration involves a restrained Amber minimization, with
    restraint exclusions determined by violation-participating residues.

    Args:
      prot: A protein to be relaxed.
      stiffness: kcal/mol A**2, the restraint stiffness.
      max_outer_iterations: The maximum number of iterative minimization.
      place_hydrogens_every_iteration: Whether hydrogens are re-initialized
          prior to every minimization.
      max_iterations: An `int` specifying the maximum number of L-BFGS steps
          per relax iteration. A value of 0 specifies no limit.
      tolerance: kcal/mol, the energy tolerance of L-BFGS.
          The default value is the OpenMM default.
      restraint_set: The set of atoms to restrain.
      max_attempts: The maximum number of minimization attempts per iteration.
      checks: Whether to perform cleaning checks.
      exclude_residues: An optional list of zero-indexed residues to exclude from
          restraints.

    Returns:
      out: A dictionary of output values.
    """

    # `protein.to_pdb` will strip any poorly-defined residues so we need to
    # perform this check before `clean_protein`.


    exclude_residues = exclude_residues or []
    exclude_residues = set(exclude_residues)
    violations = np.inf
    iteration = 0

    while violations > 0 and iteration < max_outer_iterations:
        # print("Out iteration:", iteration)
        ret = _run_one_iteration(
            pdb_string=pdb_string,
            exclude_residues=exclude_residues,
            max_iterations=max_iterations,
            tolerance=tolerance,
            stiffness=stiffness,
            restraint_set=restraint_set,
            max_attempts=max_attempts)
        prot = Protein(io.StringIO(ret["min_pdb"]))
        if place_hydrogens_every_iteration:
            pdb_string = prot.clean_protein(checks=True)
        else:
            pdb_string = ret["min_pdb"]
        mats,violations=get_violation_metrics(prot)
        ret.update({
            "num_exclusions": len(exclude_residues),
            "iteration": iteration,
        })
        iteration += 1
    return ret


def relax(prot,
          stiffness: float = 10.,
          max_outer_iterations: int = 1,
          place_hydrogens_every_iteration: bool = False,
          max_iterations: int = 0,
          tolerance: float = 2.39,
          restraint_set: str = "non_hydrogen",
          max_attempts: int = 100,
          checks: bool = True,
          exclude_residues: Optional[Sequence[int]] = None):
    _check_residues_are_well_defined(prot)
    pdb_string = prot.clean_protein(checks=checks)
    out = run_pipeline(pdb_string, stiffness=stiffness, max_outer_iterations=max_outer_iterations,
                       place_hydrogens_every_iteration=place_hydrogens_every_iteration,
                       max_iterations=max_iterations, tolerance=tolerance,
                       restraint_set=restraint_set, max_attempts=max_attempts,
                       checks=checks, exclude_residues=exclude_residues)
    min_pos = out['pos']
    start_pos = out['posinit']
    rmsd = np.sqrt(np.sum((start_pos - min_pos) ** 2) / start_pos.shape[0])
    debug_data = {
        'initial_energy': out['einit'],
        'final_energy': out['efinal'],
        'attempts': out['min_attempts'],
        'rmsd': rmsd
    }

    min_pdb = utils.overwrite_pdb_coordinates(pdb_string, min_pos)
    min_pdb = utils.overwrite_b_factors(min_pdb, prot.b_factors)
    return min_pdb