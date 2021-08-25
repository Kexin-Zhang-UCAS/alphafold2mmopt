import io
from typing import Optional
from Bio.PDB import PDBParser
import numpy as np
from . import residue_constants
from . import cleanup
from simtk.openmm.app.internal.pdbstructure import PdbStructure
import simtk.openmm.app as app
from ..utils import cost

class Protein():
    def __init__(self,fname):
        self.atom_positions=""
        self.aatype=""
        self.atom_mask=""
        self.residue_index=""
        self.b_factors=""
        self.from_pdb_file(fname)
    @cost
    def from_pdb_file(self,fname,chain_id: Optional[str] = None):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('none', fname)
        models = list(structure.get_models())
        assert len(models)==1, f'Only single model_ PDBs are supported. Found {len(models)} models.'
        model = models[0]
        if chain_id is not None:
            chain = model[chain_id]
        else:
            chains = list(model.get_chains())
            assert len(chains)==1, f'Only single chain PDBs are supported when chain_id not specified. Found {len(chains)} chains.'
            chain = chains[0]
        atom_positions = []
        aatype = []
        atom_mask = []
        residue_index = []
        b_factors = []
        for res in chain:
            if res.id[2] != ' ':
                raise ValueError(
                    f'PDB contains an insertion code at chain {chain.id} and residue '
                    f'index {res.id[1]}. These are not supported.')
            res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num)
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            b_factors.append(res_b_factors)
        self.aatype=np.array(aatype)
        self.atom_mask=np.array(atom_mask)
        self.atom_positions=np.array(atom_positions)
        self.residue_index=np.array(residue_index)
        self.b_factors=np.array(b_factors)
    def to_pdb(self):
        restypes = residue_constants.restypes + ['X']
        res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], 'UNK')
        atom_types = residue_constants.atom_types
        pdb_lines = []
        # residue_index = self.residue_index.astype(np.int32)
        if np.any(self.aatype > residue_constants.restype_num):
            raise ValueError('Invalid aatypes.')

        pdb_lines.append('MODEL     1')
        atom_index = 1
        chain_id = 'A'
        # Add all atom sites.
        for i in range(self.aatype.shape[0]):
            res_name_3 = res_1to3(self.aatype[i])
            for atom_name, pos, mask, b_factor in zip(
                    atom_types, self.atom_positions[i], self.atom_mask[i], self.b_factors[i]):
                if mask < 0.5:
                    continue

                record_type = 'ATOM'
                name = atom_name if len(atom_name) == 4 else f' {atom_name}'
                alt_loc = ''
                insertion_code = ''
                occupancy = 1.00
                element = atom_name[0]  # Protein supports only C, N, O, S, this works.
                charge = ''
                # PDB is a columnar format, every space matters here!
                atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                             f'{res_name_3:>3} {chain_id:>1}'
                             f'{self.residue_index[i]:>4}{insertion_code:>1}   '
                             f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                             f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                             f'{element:>2}{charge:>2}')
                pdb_lines.append(atom_line)
                atom_index += 1

        # Close the chain.
        chain_end = 'TER'
        chain_termination_line = (
            f'{chain_end:<6}{atom_index:>5}      {res_1to3(self.aatype[-1]):>3} '
            f'{chain_id:>1}{self.residue_index[-1]:>4}')
        pdb_lines.append(chain_termination_line)
        pdb_lines.append('ENDMDL')

        pdb_lines.append('END')
        pdb_lines.append('')
        return '\n'.join(pdb_lines)
    def ideal_atom_mask(self):
        return residue_constants.STANDARD_ATOM_MASK[self.aatype]
    def from_prediction(self):
        pass

    def _check_atom_mask_is_ideal(self):
        """Sanity-check the atom mask is ideal, up to a possible OXT."""
        # atom_mask = prot.atom_mask
        ideal_atom_mask = self.ideal_atom_mask()
        # TODO
        # utils.assert_equal_nonterminal_atom_types(atom_mask, ideal_atom_mask)
    @cost
    def clean_protein(self,checks=True):
        self._check_atom_mask_is_ideal()
        # Clean pdb.
        prot_pdb_string = self.to_pdb()
        pdb_file = io.StringIO(prot_pdb_string)
        alterations_info = {}
        fixed_pdb = cleanup.fix_pdb(pdb_file, alterations_info)
        fixed_pdb_file = io.StringIO(fixed_pdb)
        pdb_structure = PdbStructure(fixed_pdb_file)
        cleanup.clean_structure(pdb_structure, alterations_info)
        # print("alterations info: %s", alterations_info)
        # Write pdb file of cleaned structure.
        as_file = app.PDBFile(pdb_structure)
        with io.StringIO() as f:
            as_file.writeFile(as_file.getTopology(), as_file.getPositions(), f)
            pdb_string = f.getvalue()
        if checks:
            cleanup._check_cleaned_atoms(pdb_string, prot_pdb_string)
        return pdb_string






