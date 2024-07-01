import sys  # System-specific parameters and functions
import os  # Miscellaneous operating system interfaces
import io 
import time  # Time access and conversions
import random  # Generate pseudo-random numbers
import math  # Mathematical functions
from copy import deepcopy  # Create deep copies of objects
from itertools import chain, islice  # Functions creating iterators for efficient looping
from collections import defaultdict  # Dictionary subclass that calls a factory function to supply missing values
import pandas as pd  # Data manipulation and analysis (DataFrames)
import numpy as np
import scipy  # Scientific and technical computing
import itertools  # Functions creating iterators for efficient looping
from itertools import chain

#rdkit imports 
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import inchi
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.rdMolDescriptors import CalcNumRings, CalcNumAromaticRings
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdmolops

import contextlib
from mpi4py import MPI


class LogCollector:
    """
    Class to capture and suppress output streams (stdout and stderr),
    including handling system errors from C libraries.
    """
    def __init__(self, stream=None):
        self.stream = stream or sys.stderr
        self._stringio = io.StringIO()

    def begin_capture(self):
        """
        Begin capturing the stream data.
        """
        self._stringio = io.StringIO()
        self._redirector = contextlib.redirect_stderr(self._stringio) if self.stream is sys.stderr else contextlib.redirect_stdout(self._stringio)
        self._redirector.__enter__()
        print("Capture started.", file=sys.__stdout__)  # Print to the original stdout

    def end_capture(self):
        """
        End capturing the stream data and save the text in `capturedtext`.
        """
        print("Ending capture...", file=sys.__stdout__)  # Print to the original stdout
        self._redirector.__exit__(None, None, None)
        self.capturedtext = self._stringio.getvalue()
        self._stringio.close()
        print("Capture ended.", file=sys.__stdout__)  # Print to the original stdout

    def get_output(self):
        """
        Get the captured output.
        """
        return self.capturedtext

output = LogCollector(sys.stderr)

def molecule(smiles, code):
    """
    Function to create a dictionary for each molecule.
    The dictionary has the following keys:
        smiles:         smiles code of the molecule
        code:           serial number/code assigned to each molecule that contains information about the building blocks used to construct that molecule
        can_smiles:     canonical smiles of the molecule, which are used to remove duplicates
        reverse_smiles: smiles of the molecule returned from the reverse_mol function

    Parameters
    ----------
    smiles: str
        smiles string of molecule
    code: str
        code for molecule (F1, F2 ...) generated automatically

    Returns
    -------
    mol: dict
        dictionary with keys - smiles, code, can_smiles, reverse_smiles
    """
    mol = {'smiles': smiles, 'code': code}
    rdkit_mol = Chem.MolFromSmiles(smiles)
    mol['can_smiles'] = Chem.MolToSmiles(rdkit_mol, canonical=True)
    rdkit_mol_can = Chem.MolFromSmiles(mol['can_smiles'])
    mol['reverse_smiles'] = reverse_mol(rdkit_mol_can, list(rdkit_mol_can.GetAtoms()))

    return mol

def reverse_mol(mol, atoms):
    """
    Function that converts a molecule's potential reactive sites into H atoms. New molecules are generated at only those sites which have H atoms.
    If user does not provide any [Ra] handles in the molecule, all available H atom sites are considered for reaction.
    If user provides specific handles for reaction by replacing H atoms with [Ra] atoms, only those sites will be considered for reaction. For the 
    reaction, the H atoms are replaced with Fr atoms so that they are not considered as potential sites, and the Ra atoms are converted to H atoms. 

    Parameters
    ----------
    mol: object
        RDKit object of a molecule
    atoms: list
        list of atoms in the molecule

    Returns
    -------
    smiles: str
        canonical smiles of transformed molecule
    """
    
    atom_num = [atom.GetAtomicNum() for atom in atoms] # to append atomic number of atoms present in molecule
    new_mol = Chem.RWMol(mol)
    
    # Convert Ra atoms to H and H to Fr
    for atom in atoms:
        if atom.GetAtomicNum() == 88:  # Assuming Ra is represented by atomic number 88
            atom.SetAtomicNum(1)  # Convert Ra to H
        elif atom.GetAtomicNum() == 1:
            atom.SetAtomicNum(87)  # Convert H to Fr (assuming Fr is represented by atomic number 87)
    
    return Chem.MolToSmiles(new_mol, canonical=True)

def check_building_blocks(smiles, line, file_name, output_dir, mpidict):
    """
    Validate the building blocks input (smiles or inchi) and return the smiles of the molecule

    Parameters
    ----------
    smiles: str
        Molecule representation in 2D format (SMILES/InChI)
    line: int
        Line number in input file
    file_name: str
        File handle
    output_dir: str
        Output directory for logging errors
    mpidict: dict
        Dictionary for multiprocessing

    Returns
    -------
    smiles: str
        SMILES representation of molecule
    """
    inchi_bb, smiles_bb = True, True

    # Check if input is InChI
    try:
        mol = inchi.MolFromInchi(smiles)
        if mol is not None:
            smiles = Chem.MolToSmiles(mol, canonical=True)
            smiles = smiles.strip()
        else:
            inchi_bb = False
    except:
        inchi_bb = False
    
    # Check if input is SMILES
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            smiles_bb = False
    except:
        smiles_bb = False

    if inchi_bb == False and smiles_bb == False:
        tmp_str = f"Error: The SMILES/InChI string ('{smiles}') provided in line {line} of data file '{file_name}' is not valid. Please provide correct SMILES/InChI."
        log_error(tmp_str, output_dir, mpidict, "Aborting due to wrong molecule description.", error=True)
    else:
        return smiles

    return smiles

def get_rules(config_file, output_dir, mpidict):
    """ 
    Function to read generation rules provided in the config file.
        
    Parameters
    ----------
    config_file: file handle
    output_dir: str
        Output directory for logging errors
    mpidict: dict
        Dictionary for multiprocessing

    Returns
    -------
    rules_dict: dict
        Dictionary of generation rules provided by the user. If the user provides default values for any rule, it is not added to the dictionary.
    lib_args: list
        List of other input arguments to the library generator
    """
    rules_dict, lib_args = {}, []

    for i, line in enumerate(config_file):
        if i == 0:
            continue
        log_error(line[:-1], output_dir, mpidict)

        if '==' in line:
            words = line.split('==')
            value = words[1].strip()

            if value == 'None':
                continue
            
            elif i == 1:
                if not isinstance(eval(value), tuple):
                    tmp_str = "ERROR: Wrong generation rule provided for " + line
                    log_error(tmp_str, output_dir, mpidict, "Aborting due to wrong generation rule.", error=True)
                rules_dict['include_bb'] = [i.strip() for i in eval(value)]
                continue
                
            elif i == 11:
                if isinstance(eval(value), tuple):
                    rules_dict['heteroatoms'] = eval(value)
                else:
                    tmp_str = "ERROR: Wrong generation rule provided for " + line
                    log_error(tmp_str, output_dir, mpidict, "Aborting due to wrong generation rule.", error=True)
                continue
        
            elif i == 12:
                if value == "True":
                    rules_dict['lipinski'] = True
                elif value == "False":
                    continue
                else:
                    tmp_str = "ERROR: Wrong generation rule provided for " + line
                    log_error(tmp_str, output_dir, mpidict, "Aborting due to wrong generation rule.", error=True)
                continue

            elif i == 13:  # This rule is for fingerprint matching
                target_mols = value.split(',')
                smiles_to_comp = []
                for j in target_mols:
                    target_smiles, tanimoto_index = j.split('-')[0].strip(), j.split('-')[1].strip()
                    smiles = check_building_blocks(target_smiles, i + 1, config_file.name, output_dir, mpidict)
                    smiles_to_comp.append([smiles, tanimoto_index])
                rules_dict['fingerprint'] = smiles_to_comp            
                continue

            elif i == 14 or i == 15:  # This rule is for substructure inclusion and exclusion
                smiles_l = []
                for item in value.split(','):
                    smiles = check_building_blocks(item.strip(), i + 1, config_file.name, output_dir, mpidict)
                    smiles_l.append(smiles)
                rules_dict[str(i)] = smiles_l
                continue

            elif i == 16:  # This rule is for inclusion of initial building blocks in the final library
                if value != 'True' and value != 'False':
                    tmp_str = "ERROR: Wrong generation rule provided for " + line
                    tmp_str += "Provide either True or False. \n"
                    log_error(tmp_str, output_dir, mpidict, "Aborting due to wrong generation rule.", error=True)
                
                if value == 'False':
                    rules_dict['bb_final_lib'] = False
                elif value == 'True':
                    rules_dict['bb_final_lib'] = True
                continue
            
            elif value != 'None':
                if not isinstance(eval(value), tuple) or len(eval(value)) != 2:
                    tmp_str = "ERROR: Wrong generation rule provided for " + line
                    tmp_str += "Provide the range in tuple format (min, max). \n"
                    log_error(tmp_str, output_dir, mpidict, "Aborting due to wrong generation rule.", error=True)
                else:
                    rules_dict[str(i)] = eval(value)
            
        elif '::' in line:
            words = line.split('::')
            value = words[1].strip()
            lib_args.append(value)

    return rules_dict, lib_args

def parse_ga(config_file, output_dir, mpidict):
    """ 
    Function to read parameters for genetic algorithm.
        
    Parameters
    ----------
    config_file: file handle
    output_dir: str
        Output directory for logging errors
    mpidict: dict
        Dictionary for multiprocessing

    Returns
    -------
    params: list
        ['fitness', 'crossover_size', 'mutation_size', 'algorithm', 'generations', 'Initial population ratio', 'Crossover population ratio']
    """
    params = []
    param_list = ['Batch run', 'fitness', 'crossover_size', 'mutation_size', 'algorithm', 'generations', 'Initial population ratio', 'Crossover population ratio']
    lines = config_file.readlines()

    if len(lines) != len(param_list):
        log_error('Error: Incorrect config file provided for genetic algorithm', output_dir, mpidict, "Aborting due to wrong config file format.", error=True)
        return []

    for ind, line in enumerate(lines):
        if param_list[ind] not in line:
            log_error(f'Error: Missing parameter {param_list[ind]} in line {ind + 1}', output_dir, mpidict, "Aborting due to missing parameter.", error=True)
            return []

        value = line.split('=')[1].strip()
        
        if ind == 0:  # Batch run
            if value.lower() == 'false':
                params.append(False)
            elif value.lower() == 'true':
                params.append(True)
            else:
                log_error(f'Error: Incorrect option for {param_list[ind]} provided in genetic algorithm config', output_dir, mpidict, "Aborting due to incorrect option.", error=True)
                return []

        elif ind == 1:  # Fitness
            try:
                eval_value = eval(value)
                if not isinstance(eval_value, tuple):
                    log_error(f'Error: Incorrect format for fitness value provided in genetic algorithm config', output_dir, mpidict, "Aborting due to incorrect format.", error=True)
                    return []
                params.append(eval_value)
            except:
                log_error(f'Error: Unable to evaluate fitness value provided in genetic algorithm config', output_dir, mpidict, "Aborting due to evaluation error.", error=True)
                return []

        else:  # Other parameters
            try:
                eval_value = eval(value)
                if not isinstance(eval_value, (int, float)):
                    log_error(f'Error: Incorrect format for {param_list[ind]} provided in genetic algorithm config', output_dir, mpidict, "Aborting due to incorrect format.", error=True)
                    return []
                params.append(eval_value)
            except:
                log_error(f'Error: Unable to evaluate {param_list[ind]} value provided in genetic algorithm config', output_dir, mpidict, "Aborting due to evaluation error.", error=True)
                return []
    
    if None in params:
        log_error('Error: Values cannot be None in config file for genetic algorithm', output_dir, mpidict, "Aborting due to None values.", error=True)
        return []
    
    for param, name in zip(params, param_list):
        log_error(f'{name}: {param}', output_dir, mpidict)

    return params

def lipinski(mol):
    """
    Function that returns the values of the Lipinski descriptors.

    Parameters
    ----------
    mol: object
        RDKit Mol object of the molecule

    Returns
    -------
    desc: dict
        Dictionary of Lipinski descriptors
    """
    desc = {
        'molwt': Descriptors.MolWt(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'logP': Descriptors.MolLogP(mol)
    }

    return desc

def unique_structs(mol, smarts):
    """
    Function to calculate the number of given sub-structures (SMARTS) in the provided molecule.

    Parameters
    ----------
    mol: object
        RDKit Mol object of the molecule
    smarts: str
        SMARTS representation of the sub-structure

    Returns
    -------
    num_unique_matches: int
        Number of unique matches found for the given sub-structure in the molecule
    """
    substructure = Chem.MolFromSmarts(smarts)
    if substructure is None:
        raise ValueError("Invalid SMARTS pattern provided.")

    matches = mol.GetSubstructMatches(substructure, uniquify=True)
    num_unique_matches = len(matches)
    
    return num_unique_matches

def log_error(sentence, output_dir, mpidict, msg="Aborting the run", error=False):
    """
    Print to both error file and logfile and then exit code.
    
    Parameters
    ----------
    sentence: str
        The error message to log
    output_dir: str
        The directory where log files will be saved
    mpidict: dict
        Dictionary containing MPI information, e.g., rank
    msg: str, optional
        Message to print upon exiting (default is "Aborting the run")
    error: bool, optional
        Flag indicating whether this is an error message (default is False)
    """
    rank = mpidict.get('rank', 0)
    
    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logfile_path = os.path.join(output_dir, 'logfile.txt')
        error_file_path = os.path.join(output_dir, 'error_file.txt')

        with open(logfile_path, 'a') as logfile:
            print(sentence)
            logfile.write(sentence + "\n")

        if error:
            with open(error_file_path, 'a') as error_file:
                error_file.write(sentence + "\n")
            sys.exit(msg)
    else:
        if error:
            sys.exit()

def if_add(molc, rules, code, check_min=False):
    """
    Determines if a molecule should be included in the library based on the generation rules given.

    Parameters
    ----------
    molc: str 
        Reversed canonical smiles of molecule whose validity is to be determined
    rules: dict 
        Dictionary of rules for library generation 
    code: str
        Code for molecule (F1, F2 ...)

    Returns
    -------
    add: bool 
        True/False
    """
    # Check validity of molecule
    if not molc:
        return False

    try:
        rdkit_mol = Chem.MolFromSmiles(molc)
        if rdkit_mol is None:
            print("Invalid SMILES:", molc)
            return False
    except:
        print("Invalid SMILES:", molc)
        return False

    # Create a new copy of the dict to avoid changing the original dict
    mol = molecule(molc, code)
    rdkit_mol = Chem.MolFromSmiles(mol['reverse_smiles'])

    add = True

    # Removing Francium and Radium atoms
    for atom in rdkit_mol.GetAtoms():
        if atom.GetAtomicNum() == 87 or atom.GetAtomicNum() == 88:
            atom.SetAtomicNum(1)

    if 'include_bb' in rules:
        for item in rules['include_bb']:
            no_occ = unique_structs(rdkit_mol, item)
            if no_occ <= 0:
                add = False
                del mol
                return add
    
    if '2' in rules:
        bonds = rdkit_mol.GetNumBonds()
        if not bonds <= rules['2'][1]:
            add = False
            del mol
            return add
    
    if '3' in rules:
        if not len(rdkit_mol.GetAtoms()) <= rules['3'][1]:
            add = False
            del mol
            return add

    # Calculating number of rings
    if '5' in rules:
        rings = Chem.GetSSSR(rdkit_mol)
        if not rings <= rules['5'][1]:
            add = False
            del mol
            return add

    # Calculating number of aromatic and non-aromatic rings
    if '6' in rules or '7' in rules:
        no_ar = len([ring for ring in Chem.GetSymmSSSR(rdkit_mol) if Chem.rdMolDescriptors.CalcNumAromaticRings(rdkit_mol, ring.GetIdx())])
        no_non_ar = rings - no_ar
        if '6' in rules and not no_ar <= rules['6'][1]:
            add = False
            del mol
            return add
        if '7' in rules and not no_non_ar <= rules['7'][1]:
            add = False
            del mol
            return add

    if '8' in rules:
        no_s_bonds = unique_structs(rdkit_mol, "*-*")
        if not no_s_bonds <= rules['8'][1]:
            add = False
            del mol
            return add

    if '9' in rules:
        no_d_bonds = unique_structs(rdkit_mol, "*=*")
        if not no_d_bonds <= rules['9'][1]:
            add = False
            del mol
            return add

    if '10' in rules:
        no_t_bonds = unique_structs(rdkit_mol, "*#*")
        if not no_t_bonds <= rules['10'][1]:
            add = False
            del mol
            return add
    
    if 'heteroatoms' in rules:
        for item in rules['heteroatoms']:
            no_at = unique_structs(rdkit_mol, item[0])
            if no_at > item[1]:
                add = False
                del mol
                return add
    
    if 'lipinski' in rules:
        descriptors = lipinski(rdkit_mol)
        if not ((descriptors['molwt'] <= 500) and (descriptors['HBD'] <= 5) and (descriptors['HBA'] <= 10) and (descriptors['logP'] <= 5)):
            add = False
            del mol
            return add

    if 'fingerprint' in rules:
        for mol_to_comp in rules['fingerprint']:
            rdkit_mol_comp = Chem.MolFromSmiles(mol_to_comp[0])
            if rdkit_mol_comp is None:
                continue
            fp1 = Chem.RDKFingerprint(rdkit_mol)
            fp2 = Chem.RDKFingerprint(rdkit_mol_comp)
            tanimoto = Chem.DataStructs.FingerprintSimilarity(fp1, fp2)
            if tanimoto <= float(mol_to_comp[1]):
                add = False
                del mol
                return add

    if '14' in rules:
        for item in rules['14']:
            no_occ = unique_structs(rdkit_mol, item)
            if no_occ == 0:
                add = False
                del mol
                return add 
    
    if '15' in rules:
        for item in rules['15']:
            no_occ = unique_structs(rdkit_mol, item)
            if no_occ > 0:
                add = False
                del mol
                return add    

    # Check minimum criteria if flag is true
    if check_min:
        if '2' in rules:
            bonds = rdkit_mol.GetNumBonds()
            if not rules['2'][0] <= bonds:
                add = False
                del mol
                return add
        
        if '3' in rules:
            if not rules['3'][0] <= len(rdkit_mol.GetAtoms()):
                add = False
                del mol
                return add

        if '4' in rules:
            molwt = Descriptors.MolWt(rdkit_mol)
            if not rules['4'][0] <= molwt <= rules['4'][1]:
                add = False
                del mol
                return add

        # Calculating number of rings
        if '5' in rules:
            rings = Chem.GetSSSR(rdkit_mol)
            if not rules['5'][0] <= rings:
                add = False
                del mol
                return add

        # Calculating number of aromatic and non-aromatic rings
        if '6' in rules or '7' in rules:
            no_ar = len([ring for ring in Chem.GetSymmSSSR(rdkit_mol) if Chem.rdMolDescriptors.CalcNumAromaticRings(rdkit_mol, ring.GetIdx())])
            no_non_ar = rings - no_ar
            if '6' in rules and not rules['6'][0] <= no_ar:
                add = False
                del mol
                return add
            if '7' in rules and not rules['7'][0] <= no_non_ar:
                add = False
                del mol
                return add

        if '8' in rules:
            no_s_bonds = unique_structs(rdkit_mol, "*-*")
            if not rules['8'][0] <= no_s_bonds:
                add = False
                del mol
                return add

        if '9' in rules:
            no_d_bonds = unique_structs(rdkit_mol, "*=*")
            if not rules['9'][0] <= no_d_bonds:
                add = False
                del mol
                return add

        if '10' in rules:
            no_t_bonds = unique_structs(rdkit_mol, "*#*")
            if not rules['10'][0] <= no_t_bonds:
                add = False
                del mol
                return add

    del mol
    return add

def get_index_list(mol, atoms):
    """
    Returns the list of index numbers of atoms that can be reacted in a molecule while making sure no duplicate sites are returned.

    Parameters
    ----------
    mol: dict
        Dictionary object of molecule
    atoms: list
        List of atoms in the molecule

    Returns
    -------
    atoms_index: list
        List of atom indices in the molecule that can be reacted
    """
    
    can_smiles_list = []
    atoms_index = []

    for atom in atoms:
        # Counting the number of hydrogens attached to the atom. Do not do anything if there are no hydrogens attached
        hcount = atom.GetTotalNumHs()
        if hcount == 0:
            continue 
        
        newmol = Chem.MolFromSmiles(mol['reverse_smiles'])
        rw_newmol = Chem.RWMol(newmol)
        # Attach Francium atom. Makes it easy to remove duplicates
        fr_atom = Chem.Atom(87)  # Francium atom
        fr_idx = rw_newmol.AddAtom(fr_atom)
        index = atom.GetIdx()
        # Create a bond between Fr atom and the current atom
        rw_newmol.AddBond(index, fr_idx, Chem.rdchem.BondType.SINGLE)
        atom.SetNumExplicitHs(hcount - 1)
        
        # Making use of canonical smiles to remove duplicates
        can_smiles = Chem.MolToSmiles(rw_newmol, canonical=True)
        
        if can_smiles not in can_smiles_list:
            can_smiles_list.append(can_smiles)
            atoms_index.append(index)
    
    return atoms_index

def create_link(mol1, mol2, rules):
    """
    This function creates all possible links between two given molecules.

    Parameters
    ----------
    mol1: dict
        Dictionary object of first molecule
    mol2: dict
        Dictionary object of second molecule
    rules: dict
        Dictionary of rules for library generation 

    Returns
    -------
    library_two: list 
        List of molecule dict objects
    """
    library_two = []
    mol1_rdkit = Chem.MolFromSmiles(mol1['reverse_smiles'])
    mol2_rdkit = Chem.MolFromSmiles(mol2['reverse_smiles'])
    mol1_index_list = get_index_list(mol1, mol1_rdkit.GetAtoms())
    mol2_index_list = get_index_list(mol2, mol2_rdkit.GetAtoms())
    
    smiles_combi = mol1['reverse_smiles'] + '.' + mol2['reverse_smiles']
    code = mol1['code'] + '-' + mol2['code']

    for index1 in mol1_index_list:
        for index2 in mol2_index_list:
            mol_combi = Chem.CombineMols(mol1_rdkit, mol2_rdkit)
            rw_mol_combi = Chem.RWMol(mol_combi)
            rw_mol_combi.AddBond(index1, index2 + mol1_rdkit.GetNumAtoms(), Chem.rdchem.BondType.SINGLE)
            atom1 = rw_mol_combi.GetAtomWithIdx(index1)
            atom2 = rw_mol_combi.GetAtomWithIdx(index2 + mol1_rdkit.GetNumAtoms())
            atom1.SetNumExplicitHs(atom1.GetTotalNumHs() - 1)
            atom2.SetNumExplicitHs(atom2.GetTotalNumHs() - 1)
            Chem.SanitizeMol(rw_mol_combi)
            can_mol_combi = Chem.MolToSmiles(rw_mol_combi, canonical=True)
            if if_add(can_mol_combi, rules, code):
                temp = molecule(can_mol_combi, code)
                library_two.append(temp)
    return library_two

def get_fusion_index(molc, mol_type):

    """Function to get list of all potential sites for fusion after removing duplicate sites.

    Parameters
    ----------
    molc: dict
        Dictionary object of molecule
    mol_type: int
        Values = (1,2). 1 indicates the molecule from which no carbon atoms are deleted. 
        2 indicates molecule which loses 2 carbon atoms so that the immediate next 
        neighbors of these 2 atoms can be attached to the first molecule.

    Returns
    -------
    atom_pair_list: list
        For mol_type 1 - list of index of [atom, alpha/neighboring atom]
        For mol_type 2 - list of index of [atom, 1st alpha atom, 2nd alpha atom, beta atom for 1st alpha atom]
    len(atoms): int
        Number of atoms in the molecule
    substitutions: dict
        Dictionary that contains the side chain substitutions on the 2 atoms that will fuse.
    """
    def modify_smiles(rdkit_mol, a1, a2):
        rw_mol = Chem.RWMol(rdkit_mol)
        # Attach Francium atoms for duplicate removal
        fr_atom1 = Chem.Atom(87)
        fr_atom2 = Chem.Atom(87)
        fr_idx1 = rw_mol.AddAtom(fr_atom1)
        fr_idx2 = rw_mol.AddAtom(fr_atom2)
        rw_mol.AddBond(a1, fr_idx1, Chem.rdchem.BondType.SINGLE)
        rw_mol.AddBond(a2, fr_idx2, Chem.rdchem.BondType.SINGLE)
        atom1 = rw_mol.GetAtomWithIdx(a1)
        atom2 = rw_mol.GetAtomWithIdx(a2)
        atom1.SetNumExplicitHs(atom1.GetTotalNumHs() - 1)
        atom2.SetNumExplicitHs(atom2.GetTotalNumHs() - 1)
        Chem.SanitizeMol(rw_mol)
        can_smiles = Chem.MolToSmiles(rw_mol, canonical=True)
        return can_smiles

    rdkit_mol = Chem.MolFromSmiles(molc['reverse_smiles'])
    atoms = rdkit_mol.GetAtoms()
    atom_pair_list, can_list, substitutions = [], [], defaultdict(list)

    for atom in atoms:
        index = atom.GetIdx()
        hcount = atom.GetTotalNumHs()
        if hcount == 0 or not atom.IsInRing():
            continue
        # look for 1st alpha atom
        for neighbor in atom.GetNeighbors():
            neighbor_hcount = neighbor.GetTotalNumHs()
            alpha_1 = neighbor.GetIdx()
            if neighbor_hcount == 0 or not neighbor.IsInRing():
                if not neighbor.IsInRing(): 
                    if alpha_1 not in substitutions[index]:
                        substitutions[index].append(alpha_1)
                continue
            alpha_1 = neighbor.GetIdx()
            
            if mol_type == 1:
                modified = modify_smiles(rdkit_mol, index, alpha_1)
                if modified not in can_list:
                    can_list.append(modified)
                    atom_pair_list.append([index, alpha_1])

            if mol_type == 2:
                beta_1 = None
                # look for beta atom
                for second_neighbor in neighbor.GetNeighbors():
                    beta_1 = second_neighbor.GetIdx()
                    if not second_neighbor.IsInRing() or second_neighbor.GetIdx() == index: 
                        if not second_neighbor.IsInRing(): 
                            if beta_1 not in substitutions[alpha_1]:
                                substitutions[alpha_1].append(beta_1)
                        continue
                    break

                alpha_2 = None
                # look for 2nd alpha atom
                for second_neighbor in atom.GetNeighbors():
                    if not second_neighbor.IsInRing() or second_neighbor.GetIdx() == alpha_1:
                        continue
                    alpha_2 = second_neighbor.GetIdx()
                    break

                if beta_1 is not None and alpha_2 is not None:
                    modified = modify_smiles(rdkit_mol, index, alpha_1)
                    if modified not in can_list:
                        can_list.append(modified)
                        atom_pair_list.append([index, alpha_1, alpha_2, beta_1])                  

    return atom_pair_list, len(atoms), substitutions

def create_fused(mol1, mol2, rules):
    """Function that returns the list of all molecules resulting from fusion of two molecules.

    Parameters
    ----------
    mol1: dict
        Molecule dictionary object
    mol2: dict
        Molecule dictionary object
    rules: dict
        Dictionary of generation rules

    Returns
    -------
    library_two: list
        List of all possible fused molecules obtained from the two given molecules 
    """
    mol1_index_list, size1, subs_1 = get_fusion_index(mol1, 1)
    mol2_index_list, size2, subs_2 = get_fusion_index(mol2, 2)
    smiles_combi = mol1['reverse_smiles'] + '.' + mol2['reverse_smiles']
    library_two = []
    code = mol1['code'] + ':' + mol2['code']
    mol_combi = Chem.MolFromSmiles(smiles_combi)
    
    for item1 in mol1_index_list:
        for item2 in mol2_index_list:
            prime_subs, alpha_subs = subs_1[item1[0]] + subs_2[item2[0]], subs_1[item1[1]] + subs_2[item2[1]]
            rw_mol_combi = Chem.RWMol(mol_combi)
            
            prime_atom = rw_mol_combi.GetAtomWithIdx(item1[0])
            alpha_1_atom = rw_mol_combi.GetAtomWithIdx(item1[1])
            alpha_2_atom = rw_mol_combi.GetAtomWithIdx(size1 + item2[2])
            beta_1_atom = rw_mol_combi.GetAtomWithIdx(size1 + item2[3])

            # check for atomic numbers of the two atoms being fused. If not same, then change it to a non-carbon atomic number
            if prime_atom.GetAtomicNum() != rw_mol_combi.GetAtomWithIdx(item2[0] + size1).GetAtomicNum():
                prime_atom.SetAtomicNum(list(set([prime_atom.GetAtomicNum(), rw_mol_combi.GetAtomWithIdx(item2[0] + size1).GetAtomicNum()]) - set([6]))[0])

            if alpha_1_atom.GetAtomicNum() != rw_mol_combi.GetAtomWithIdx(item2[1] + size1).GetAtomicNum():
                alpha_1_atom.SetAtomicNum(list(set([alpha_1_atom.GetAtomicNum(), rw_mol_combi.GetAtomWithIdx(item2[1] + size1).GetAtomicNum()]) - set([6]))[0])

            # check for aromaticity of the two molecules
            first_mol_aromatic, second_mol_aromatic = False, False
            if alpha_1_atom.GetIsAromatic() and prime_atom.GetIsAromatic(): first_mol_aromatic = True
            if alpha_2_atom.GetIsAromatic() and beta_1_atom.GetIsAromatic(): second_mol_aromatic = True

            # check for substitutions/side-chains in the two molecules and based on valency of the fusing atoms, check feasibility of fusion
            if (first_mol_aromatic or second_mol_aromatic) and (len(prime_subs) + len(alpha_subs) > 0):
                return library_two
            if not (first_mol_aromatic or second_mol_aromatic) and (len(prime_subs) > 1 or len(alpha_subs) > 1):
                return library_two
            
            # add bonds between first and second molecule, and attach side-chains of 2nd molecule to the 1st molecule
            rw_mol_combi.AddBond(item1[0], item2[2] + size1, Chem.rdchem.BondType.SINGLE)
            atom1 = rw_mol_combi.GetAtomWithIdx(item1[0])
            atom2 = rw_mol_combi.GetAtomWithIdx(item2[2] + size1)
            atom1.SetNumExplicitHs(atom1.GetTotalNumHs() - 1)
            atom2.SetNumExplicitHs(atom2.GetTotalNumHs() - 1)

            rw_mol_combi.AddBond(item1[1], item2[3] + size1, Chem.rdchem.BondType.SINGLE)
            atom1 = rw_mol_combi.GetAtomWithIdx(item1[1])
            atom2 = rw_mol_combi.GetAtomWithIdx(item2[3] + size1)
            atom1.SetNumExplicitHs(atom1.GetTotalNumHs() - 1)
            atom2.SetNumExplicitHs(atom2.GetTotalNumHs() - 1)

            for subs_ind in subs_2[item2[0]]:
                rw_mol_combi.AddBond(item1[0], subs_ind + size1, Chem.rdchem.BondType.SINGLE)
                atom1 = rw_mol_combi.GetAtomWithIdx(item1[0])
                atom2 = rw_mol_combi.GetAtomWithIdx(subs_ind + size1)
                atom1.SetNumExplicitHs(atom1.GetTotalNumHs() - 1)
                atom2.SetNumExplicitHs(atom2.GetTotalNumHs() - 1)

            for subs_ind in subs_2[item2[1]]:
                rw_mol_combi.AddBond(item1[1], subs_ind + size1, Chem.rdchem.BondType.SINGLE)
                atom1 = rw_mol_combi.GetAtomWithIdx(item1[1])
                atom2 = rw_mol_combi.GetAtomWithIdx(subs_ind + size1)
                atom1.SetNumExplicitHs(atom1.GetTotalNumHs() - 1)
                atom2.SetNumExplicitHs(atom2.GetTotalNumHs() - 1)
            
            # delete the fusing atoms from the second molecule
            atoms_to_delete = [size1 + item2[0], size1 + item2[1]]
            for i in sorted(atoms_to_delete, reverse=True):
                rw_mol_combi.RemoveAtom(i)

            # adjust aromaticity
            if second_mol_aromatic:
                rw_mol_combi.GetAtomWithIdx(item1[0]).SetIsAromatic(True)
                rw_mol_combi.GetAtomWithIdx(item1[1]).SetIsAromatic(True)
                for atom in rw_mol_combi.GetAtoms():
                    if atom.GetIdx() >= size1:
                        atom.SetIsAromatic(True)

            Chem.SanitizeMol(rw_mol_combi)
            can_mol_combi = Chem.MolToSmiles(rw_mol_combi, canonical=True)
            
            if if_add(can_mol_combi, rules, code):
                temp = molecule(can_mol_combi, code)
                library_two.append(temp)
    
    return library_two

def generator(init_mol_list, combi_type, gen_len, rules_dict, output_dir, mpidict):
    """
    Function that creates a new generation of molecules with the initial building blocks provided and the current generation of molecules.

    Parameters
    ----------
    init_mol_list: list
        List of input molecules (dict objects) with duplicates removed
    combi_type: str
        Type of combination (both, link, fusion)
    gen_len: int
        Total number of generations for which to run the library
    rules_dict: dict
        Dictionary of generation rules
    output_dir: str
        Directory for output files
    mpidict: dict
        Dictionary containing MPI communicator and rank information
    
    Returns
    -------
    library[-1]: list
        Final molecular library of dict objects after all the generations
    """
    comm, rank, mpisize = [mpidict[i] for i in mpidict]
    
    library = []
    library.append(init_mol_list)
    for gen in range(gen_len):
        log_error("\nGeneration " + str(gen+1), output_dir, mpidict)
        prev_gen_mol_list = library[gen]
        lib_temp, new_gen_mol_list = [], []
        
        chunks_list = scipy.array_split(range(len(prev_gen_mol_list)), mpisize)
        if len(chunks_list[rank]) > 0:
            for i in chunks_list[rank]:                       
                for mol2 in init_mol_list:
                    if combi_type.lower() == 'both':
                        new_gen_mol_list += create_link(prev_gen_mol_list[i], mol2, rules_dict)
                        new_gen_mol_list += create_fused(prev_gen_mol_list[i], mol2, rules_dict)
                    
                    elif combi_type.lower() == 'link':
                        new_gen_mol_list += create_link(prev_gen_mol_list[i], mol2, rules_dict)
                    
                    elif combi_type.lower() == 'fusion':
                        new_gen_mol_list += create_fused(prev_gen_mol_list[i], mol2, rules_dict)
        
        if comm is not None: 
            new_gen_mol_list = comm.gather(new_gen_mol_list, root=0)
            if rank == 0:
                new_gen_mol_list = list(chain.from_iterable(new_gen_mol_list))  # flatten out the list
            new_gen_mol_list = comm.bcast(new_gen_mol_list, root=0)

        # runs only for the last generation
        if gen == gen_len-1:
            if not rules_dict.get('bb_final_lib', True): 
                library = library[1:]
            list_to_scatter = list(chain.from_iterable(library))  # flatten out the list
            new_gen_mol_list = list_to_scatter + new_gen_mol_list  # add the last generation to library
            chunks_list = scipy.array_split(range(len(new_gen_mol_list)), mpisize)
            for i in chunks_list[rank]:
                mol_rdkit = Chem.MolFromSmiles(new_gen_mol_list[i]['reverse_smiles'])
                for atom in mol_rdkit.GetAtoms():
                    # Removing Francium and Radium atoms. It is easy to convert Francium atom to hydrogen than deleting the atom
                    if atom.GetAtomicNum() == 87 or atom.GetAtomicNum() == 88:
                        atom.SetAtomicNum(1)
                new_gen_mol_list[i]['reverse_smiles'] = Chem.MolToSmiles(mol_rdkit, canonical=True).strip()
        
        # Creating a dictionary of molecules to prevent duplicates
        smiles_dict = defaultdict(list) 
        duplicates = 0
        for l2 in new_gen_mol_list:
            mol_rdkit = Chem.MolFromSmiles(l2['reverse_smiles'])
            mol_wt = int(Chem.Descriptors.MolWt(mol_rdkit))
            smiles_dict[mol_wt].append(l2)  # appending dicts of molecule in dictionary
        
        items = list(smiles_dict.items())
        chunks_list = scipy.array_split(range(len(items)), mpisize)
        for items_ind in chunks_list[rank]:
            mol_wt, mols = items[items_ind][0], items[items_ind][1]  # mols --> list of molecules in that mol_wt category
            tmp_list = []
            for i in mols:
                if i['reverse_smiles'] not in tmp_list:
                    if gen == gen_len - 1:
                        if not if_add(i['reverse_smiles'], rules_dict, "min", check_min=True):
                            continue
                    tmp_list.append(i['reverse_smiles'])
                    lib_temp.append(i)
                else:
                    duplicates += 1
        
        if comm is not None:
            lib_temp = comm.gather(lib_temp, root=0)
            duplicates = comm.gather(duplicates, root=0)
        
        if rank == 0:
            if comm is not None:
                lib_temp = list(chain.from_iterable(lib_temp))
            library.append(lib_temp)
            log_error('Total molecules generated: ' + str(len(lib_temp)), output_dir, mpidict)
            if isinstance(duplicates, list): 
                duplicates = sum(duplicates)
            log_error('Duplicates removed: ' + str(duplicates) + '\n\n', output_dir, mpidict)
        
        if comm is not None:
            library = comm.bcast(library, root=0)
    
    return library[-1]


    """
    Log errors and optionally print user message.

    Parameters
    ----------
    message: str
        Detailed error message to log
    output_dir: str
        Directory to save the log file
    mpidict: dict
        Dictionary for multiprocessing
    user_message: str
        Message to print for the user
    error: bool
        Flag to indicate if this is an error
    """
    log_file = os.path.join(output_dir, 'error_log.txt')
    with open(log_file, 'a') as f:
        f.write(message + '\n')
    if error and user_message:
        print(user_message)

def library_generator(config_file='config.dat', building_blocks_file='building_blocks.dat', output_dir='./', genetic_algorithm_config=None, cost_function=None, fitnesses_list=None):
    """Main wrapper function for library generation.
    Generates the library based on the two input files: building_blocks.dat and config.dat
    Output: 
        Creates a csv file containing the smiles and the corresponding molecule codes.
        Creates separate files based on the output format.

    Parameters
    ----------
    config_file: str, default = 'config.dat'
        Path to the config file for generating the library
    building_blocks_file: str, default = './'
        Path to the building blocks file for generating the library.
    output_dir: str, default = './'
        Path to the output directory.
    genetic_algorithm_config: str, default = None
        Required to run genetic algorithm. Path to the config file for genetic algorithm
    cost_function: function, default = None
        Required to run genetic algorithm. Cost function that will be optimized by genetic algorithm. Cost function may return more than one value for optimization. 
    fitnesses_list: list of tuples, default = None
        Required to run genetic algorithm (GA) in batch mode. Provide empty list if running batch mode for the first time. Else, provide fitness values of individuals STRICTLY FOLLOWING THE FORMAT: [(individual generated by GA code, fitness value, smiles string of individual), ....]
    """
    # initializing MPI
    try:
        wt1 = MPI.Wtime()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        mpisize = comm.Get_size()
        mpidict = {'comm': comm, 'rank': rank, 'mpisize': mpisize}
    except: 
        wt1 = time.time()
        comm = None
        rank = 0
        mpisize = 1
        mpidict = {'comm': comm, 'rank': rank, 'mpisize': mpisize}
        log_error("Warning: MPI4PY not found. Running in serial mode.\n\n", output_dir, mpidict)
    
    log_error("Starting library generation", output_dir, mpidict)
    
    try:
        rulesFile = open(config_file)
    except:
        tmp_str = "Config file does not exist. Please provide correct config file.\n"
        log_error(tmp_str, output_dir, mpidict, "Aborting due to missing config file.", error=True)
    log_error("Reading generation rules", output_dir, mpidict)
    rules_dict, args = get_rules(rulesFile, output_dir, mpidict)
    BB_file = building_blocks_file
    combi_type, gen_len, outfile_type, max_fpf, lib_name = args
    gen_len, max_fpf = int(gen_len), int(max_fpf)
    if gen_len == 0:
        rules_dict['bb_final_lib'] = True

    # Reading the building blocks from the input file
    initial_mols = []
    log_error("Reading building blocks from the file " + BB_file, output_dir, mpidict)
    try:
        infile = open(BB_file)
    except:
        tmp_str = "Building blocks file " + BB_file + " does not exist. Please provide correct building blocks file.\n"
        log_error(tmp_str, output_dir, mpidict, "Aborting due to missing building blocks file.", error=True)
    
    i_smi_list = []
    for i, line in enumerate(infile):
        smiles = line.strip()
        if smiles.isspace() or len(smiles) == 0 or smiles[0] == '#':
            continue
        if '[X]' in smiles:
            smiles = smiles.replace('[X]', '[Ra]')
        smiles = check_building_blocks(smiles, i+1, BB_file, output_dir, mpidict)
        # removing duplicates in the input list based on canonical smiles
        temp = molecule(smiles, 'F' + str(len(initial_mols) + 1))
        is_duplicate = False
        for z in initial_mols:
            if temp['can_smiles'] not in z['can_smiles']:
                continue
            is_duplicate = True
        if not is_duplicate:
            initial_mols.append(temp)
            i_smi_list.append(temp['can_smiles'])

    log_error('Number of building blocks provided = ' + str(len(initial_mols)) + '\n', output_dir, mpidict)
    log_error('unique SMILES: ', output_dir, mpidict)
    log_error(i_smi_list, output_dir, mpidict)

    # run genetic algorithm
    if genetic_algorithm_config is not None:
        if cost_function is None and fitnesses_list is None:
            log_error("Missing input for genetic algorithm. Provide either cost function or fitnesses_list. Aborting", output_dir, mpidict, error=True)
        if cost_function is not None and fitnesses_list is not None:
            log_error("Cannot provide both cost function and fitnesses. Decide whether you wish to run genetic algorithm in batch mode (provide fitnesses_list) or continuous (provide cost function). Aborting", output_dir, mpidict, error=True)
        if mpidict['mpisize'] > 1:
            log_error("Running genetic algorithm parallel on multiple cores. THIS IS NOT REQUIRED! Instead, parallelize the cost function only.", output_dir, mpidict)
        try:
            ga_config = open(genetic_algorithm_config)
        except:
            tmp_str = "Config file for genetic algorithm does not exist. Please provide correct config file.\n"
            log_error(tmp_str, output_dir, mpidict, "Aborting due to missing config file.", error=True)
        log_error("Reading parameters for genetic algorithm", output_dir, mpidict)
        batch, fitness, crossover_size, mutation_size, algorithm, generations, init_ratio, crossover_ratio = parse_ga(ga_config, output_dir, mpidict)
        
        ga_libgen = GeneticAlgorithm(evaluate=cost_function,
                                     fitness=fitness,
                                     crossover_size=crossover_size,
                                     mutation_size=mutation_size,
                                     algorithm=algorithm,
                                     initial_mols=initial_mols,
                                     rules_dict=rules_dict,
                                     output_dir=output_dir,
                                     mpidict=mpidict)

        if not batch:
            if cost_function is None:
                log_error("Missing input for genetic algorithm. Provide cost function. Aborting", output_dir, mpidict, error=True)
            ga_libgen.search(n_generations=generations, init_ratio=init_ratio, crossover_ratio=crossover_ratio)
        else:
            if fitnesses_list is None:
                log_error("Missing input for genetic algorithm. Provide fitnesses_list. Aborting", output_dir, mpidict, error=True)
            ga_libgen.batch(fitnesses_list)
        
        return ga_libgen

    # Generate molecules
    log_error('Generating molecules', output_dir, mpidict)
    final_list = generator(init_mol_list=initial_mols, combi_type=combi_type, gen_len=gen_len, rules_dict=rules_dict, output_dir=output_dir, mpidict=mpidict)
    log_error('Total number of molecules generated = ' + str(len(final_list)) + '\n', output_dir, mpidict)

    # Generating csv file of final library
    df_final_list = pd.DataFrame(final_list)
    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df_final_list.drop(['smiles', 'can_smiles'], axis=1).to_csv(os.path.join(output_dir + 'final_library.csv'), index=None)

    # Generating output files based on output file type
    if outfile_type == 'smi':
        if rank == 0:
            if not os.path.exists(output_dir + lib_name + '_' + outfile_type):
                os.makedirs(output_dir + lib_name + '_' + outfile_type)
            outdata = output_dir + lib_name + '_' + outfile_type + "/final_smiles.csv"
            log_error('Writing SMILES to file \'' + outdata + '\'\n', output_dir, mpidict)
            df_new = df_final_list['reverse_smiles'].copy()
            df_new.to_csv(outdata, index=False, header=False)
        
    # Creating a separate output file for each molecule. Files are written to folder with specified no. of files per folder.
    else:
        log_error('Writing molecules with molecule type ' + str(outfile_type) + '\n', output_dir, mpidict)
        smiles_to_scatter = []
        if rank == 0:
            if not os.path.exists(output_dir + lib_name + '_' + outfile_type):
                os.makedirs(output_dir + lib_name + '_' + outfile_type)
            smiles_to_scatter = []
            for i in range(mpisize):
                start = int(i * (len(final_list)) / mpisize)
                end = int((i + 1) * (len(final_list)) / mpisize) - 1
                list_to_add = final_list[start:end + 1]
                list_to_add = list_to_add + [len(final_list), start, end]
                smiles_to_scatter.append(list_to_add)
        else:
            smiles_to_scatter = []
    
        # Dividing the list into processors
        smiles_list = comm.scatter(smiles_to_scatter, root=0)
        final_list_len = smiles_list[-3]
        start = smiles_list[-2]
        end = smiles_list[-1]
        smiles_list = smiles_list[0:-3]
        ratio_s = int(start / max_fpf)
        ratio_e = int(end / max_fpf)
        if end + 1 == final_list_len:
            ratio_e = ratio_e + 1
        for i in range(ratio_s, ratio_e):
            if not os.path.exists(output_dir + lib_name + '_' + outfile_type + "/" + str(i + 1) + "_" + str(max_fpf)):
                os.makedirs(output_dir + lib_name + '_' + outfile_type + "/" + str(i + 1) + "_" + str(max_fpf))

        folder_no = ratio_s + 1
        for i, val in enumerate(range(start, end + 1)):
            mol_rdkit = Chem.MolFromSmiles(smiles_list[i]['reverse_smiles'])
            Chem.AllChem.EmbedMolecule(mol_rdkit)
            Chem.AllChem.MMFFOptimizeMolecule(mol_rdkit)
            mol_block = Chem.MolToMolBlock(mol_rdkit)
            with open(output_dir + lib_name + '_' + outfile_type + "/" + str(folder_no) + "_" + str(max_fpf) + "/" + str(val + 1) + "." + outfile_type, 'w') as out:
                out.write(mol_block)
            if (val + 1) % max_fpf == 0:
                folder_no = folder_no + 1
        
    log_error('File writing terminated successfully.' + '\n', output_dir, mpidict)
    if comm is not None:
        wt2 = MPI.Wtime()
    else:
        wt2 = time.time()
    log_error('Total time taken: ' + str('%.3g' % (wt2 - wt1)) + '\n', output_dir, mpidict)
    return None

class BuildingBlock(object):
    """A class for each of the building blocks that are read from the building blocks file.
    Class variables are created for smiles, molecule code, number of atoms, list of indices of potential sites for reaction, and the number of such sites.

    Parameters
    ----------
    mol: dict
        dict object created from the molecule function
    """
    def __init__(self, mol):
        self.smiles = mol['reverse_smiles']
        self.smiles_struct = mol['code']
        mol_rdkit = Chem.MolFromSmiles(mol['reverse_smiles'])
        self.atom_len = len(mol_rdkit.GetAtoms())
        self.index_list = get_index_list(mol, mol_rdkit.GetAtoms())
        self.spaces = len(self.index_list)

class GeneticAlgorithm(object):
    """
    A genetic algorithm class for search or optimization problems, built on top of the
    Distributed Evolutionary Algorithms in Python (DEAP) library. There are three algorithms with different genetic
    algorithm selection methods.

    Parameters
    ----------
    evaluate: function
        The objective function that has to be optimized. The first argument of the function is an RDKit molecule object. Objective function should return a tuple of desired target properties.

    fitness: tuple of tuple(s),
        A tuple of tuples for describing desired target properties that are returned by the objective function. For each target property, tuple contains 2 values, the required optima and the cutoff value. For 'max' optima, the cutoff is the lower acceptable limit. For 'min' optima, the cutoff is the maximum allowed value for that property. Ex: (('max', 5.6), ('min', 20))

    initial_mols: list,
        List of initial molecules.

    rules_dict: dict,
        Dictionary of generation rules.

    output_dir: str, default = './'
        Path to the output directory.

    crossover_size: int, optional (default = 50)
        Size of crossover population.

    mutation_size: int, optional (default = 50)
        Size of mutation population. Sum of crossover and mutation population is the total size of population generated in each generation.

    algorithm: int, optional (default=1)
        The algorithm to use for the search.
    """

    def __init__(self, 
                evaluate,
                fitness,
                crossover_size,
                mutation_size,
                algorithm,
                initial_mols,
                rules_dict,
                output_dir,
                mpidict):

        self.evaluate = evaluate
        self.algo = algorithm
        self.population, self.fit_list = None, ()
        self.crossover_size = int(crossover_size)
        self.mutation_size = int(mutation_size)
        self.fit_val = []
        self.pop_size = self.crossover_size + self.mutation_size
        self.output_dir = output_dir
        for i in fitness:
            if i[1] == 0:
                log_error("Cutoff values in the fitness cannot be zero.", output_dir, mpidict, error=True)
            if i[0].lower() == 'max': self.fit_val.append((1, i[1]))
            else: self.fit_val.append((-1, i[1]))

        # create building blocks class objects for each validated molecule and store them in a list
        self.bb = [BuildingBlock(i) for i in initial_mols]
        self.rules_dict = rules_dict

    def pop_generator(self, n):
        pop = []
        for _ in range(n):
            pop.append(tuple(self.chromosome_generator()))
        return pop

    def chromosome_generator(self):
        """Generates the chromosome for the algorithm, after reading and validating the molecules from the building blocks file. 

        Parameters
        ----------

        Returns
        -------
        """
        i = 0
        chromosome = []
        ind_len = random.randint(2, 5)
        while i < ind_len:
            if i == 0:
                r = random.randint(0, len(self.bb) - 1)  # randomly select building block
                chromosome.append(self.bb[r].smiles_struct)
                for j in range(self.bb[r].spaces):
                    chromosome.append([])
            else:
                avl_pos = count_list(chromosome)[0]
                if len(avl_pos) <= 0:
                    return chromosome
                r = random.randint(0, len(avl_pos) - 1)  # random number for selecting handle of 1st bb
                s = random.randint(0, len(self.bb) - 1)  # random number for selecting bb
                t = random.randint(1, self.bb[s].spaces)  # random number for selecting handle of 2nd bb
                nested_lookup(chromosome, avl_pos[r]).append(self.bb[s].smiles_struct)
                for j in range(self.bb[s].spaces):
                    if (j+1) != t:
                        nested_lookup(chromosome, avl_pos[r]).append([])
                    else:
                        nested_lookup(chromosome, avl_pos[r]).append(['C'])
            i += 1

        return deepcopy(chromosome)

    def list_to_smiles(self, indi_list):
        """The function converts the lists of lists generated by the algorithm to actual molecules.

        Parameters
        ----------
        indi_list: list,
            individual received from the algorithm.

        Returns
        -------
        mol_combi: str,
            canonical smiles of the molecule
        """
        mol_code_list, parent_list, handle_list, mol_combi, mol_len = [], [], [], '', 0
        f_lists = count_list(indi_list)[1]

        # parent list: [list, sublist, index of sublist in list]
        parent_list.append([indi_list, indi_list, -100])
        while len(parent_list) != 0:
            # iterate over new sublist
            iterate_over = parent_list[-1][1]
            new_item = False
            for k_ind, k in enumerate(iterate_over):
                # continue for loop if item already traversed
                if k_ind <= parent_list[-1][2]: continue

                if not isinstance(k, list):
                    for mol_objs in self.bb:
                        if mol_objs.smiles_struct == k:
                            new_mol_smi = mol_objs.smiles
                            mol_len += mol_objs.atom_len
                            break
                    mol_combi = mol_combi + new_mol_smi + '.'
                    if iterate_over == indi_list: nested_ind = -50
                    else:
                        iterate_over.insert(0, 'AB')
                        for fl in f_lists:
                            if nested_lookup(indi_list, fl) == iterate_over:
                                nested_ind = fl
                                break
                        del iterate_over[0]
                    # mol_code_list: [current sublist's molecule code, cumulative sum of atoms of all building blocks encountered so far, nested indices of current sublist]
                    mol_code_list.append((k, mol_len, nested_ind))
                else:
                    if k:
                        if k[0] == 'C':
                            handle_2 = k_ind
                            handle_1 = parent_list[-2][2]
                            for mol_objs in self.bb:
                                if mol_objs.smiles_struct == parent_list[-1][0][0]:
                                    handle_1 = mol_objs.index_list[handle_1 - 1]
                                if mol_objs.smiles_struct == parent_list[-1][1][0]:
                                    handle_2 = mol_objs.index_list[handle_2 - 1]

                            # get the index numbers for both handles by checking the molecule codes and their list indices.
                            for ind, mcl in enumerate(mol_code_list):
                                if mcl[2] == -50:
                                    continue
                                if mcl[0] == parent_list[-1][0][0]:
                                    parent_list[-1][0][0] += 'WW'
                                    if 'WW' in nested_lookup(indi_list, mcl[2])[0]:
                                        handle_1 += mol_code_list[ind-1][1]
                                    parent_list[-1][0][0] = parent_list[-1][0][0][:-2]

                                if mcl[0] == parent_list[-1][1][0]:
                                    parent_list[-1][1][0] += 'XX'
                                    if 'XX' in nested_lookup(indi_list, mcl[2])[0]:
                                        handle_2 += mol_code_list[ind-1][1]
                                    parent_list[-1][1][0] = parent_list[-1][1][0][:-2]

                            # append the handle indices to a list
                            handle_list.append([handle_1, handle_2])
                        else:
                            parent_list[-1][2] = k_ind
                            parent_list.append([iterate_over, k, -100])
                            new_item = True
                            break
            if not new_item:
                del parent_list[-1]

        # read the collected smiles into RDKit
        mol_combi = Chem.MolFromSmiles(mol_combi)
        rw_mol_combi = Chem.RWMol(mol_combi)
        # create bonds for each of the handles in handle_list
        for handles in handle_list:
            rw_mol_combi.AddBond(handles[0], handles[1], Chem.rdchem.BondType.SINGLE)
            x = rw_mol_combi.GetAtomWithIdx(handles[0]).GetTotalNumHs() - 1
            y = rw_mol_combi.GetAtomWithIdx(handles[1]).GetTotalNumHs() - 1
            if x >= 0: rw_mol_combi.GetAtomWithIdx(handles[0]).SetNumExplicitHs(x)
            if y >= 0: rw_mol_combi.GetAtomWithIdx(handles[1]).SetNumExplicitHs(y)

        for atom in rw_mol_combi.GetAtoms():
            ## Removing Francium and Radium atoms
            if atom.GetAtomicNum() == 87 or atom.GetAtomicNum() == 88:
                atom.SetAtomicNum(1)
        Chem.SanitizeMol(rw_mol_combi)
        mol_combi = Chem.MolToSmiles(rw_mol_combi, canonical=True)
        return mol_combi

    def pre_eval(self, indi_list):
        """Pre-processes the individuals/chromosomes before sending them to the evaluate function. 

        Parameters
        ----------
        indi_list: list,
            individual received from the algorithm.

        Returns
        -------
        fit_val: float,
            fitness value of the individual

        """
        mol_combi_smiles = self.list_to_smiles(deepcopy(list(indi_list)))
        mol_combi = Chem.MolFromSmiles(mol_combi_smiles)

        if not if_add(mol_combi_smiles, self.rules_dict, code='a'):
            return mol_combi_smiles, None
        else:
            fit_val = self.evaluate(mol_combi)
            if isinstance(fit_val, (tuple, list)): return mol_combi_smiles, tuple(fit_val)
            else: return mol_combi_smiles, tuple([fit_val])

    def crossover(self, child1, child2):
        child1, child2 = deepcopy(list(child1)), deepcopy(list(child2))
        c1 = count_list(child1)[1]
        c2 = count_list(child2)[1]
        if not (len(c1) == 0 or len(c2) == 0):
            r1 = random.randint(0, len(c1) - 1)
            r2 = random.randint(0, len(c2) - 1)
            t1 = nested_lookup(child1, c1[r1])  # pick and store a list randomly from child 1
            t2 = nested_lookup(child2, c2[r2])  # pick and store a list randomly from child 2
            if len(c1[r1]) > 1:
                del nested_lookup(child1, c1[r1][:-1])[c1[r1][-1]]
                nested_lookup(child1, c1[r1][:-1]).insert(c1[r1][-1], deepcopy(t2))
            else:
                del child1[c1[r1][0]]
                child1.insert(c1[r1][0], deepcopy(t2))

            if len(c2[r2]) > 1:
                del nested_lookup(child2, c2[r2][:-1])[c2[r2][-1]]
                nested_lookup(child2, c2[r2][:-1]).insert(c2[r2][-1], deepcopy(t1))
            else:
                del child2[c2[r2][0]]
                child2.insert(c2[r2][0], deepcopy(t1))
        return tuple(deepcopy(child1)), tuple(deepcopy(child2))

    def custom_mutate(self, indi):
        indi = deepcopy(list(indi))
        t = ['add', 'del', 'replace']
        random.shuffle(t)
        for i in t:
            if i == 'add':  # add a block randomly
                c = count_list(indi)[0]
                if not c:
                    continue
                else:
                    r = random.randint(0, len(c) - 1)  # random number for selecting empty handle
                    s = random.randint(0, len(self.bb)-1)  # random number for selecting which bb to insert
                    t = random.randint(1, self.bb[s].spaces)  # random number for selecting which handle to connect to in the new bb
                    nested_lookup(indi, c[r]).append(self.bb[s].smiles_struct)
                    for j in range(self.bb[s].spaces):
                        if (j+1) != t:
                            nested_lookup(indi, c[r]).append([])
                        else:
                            nested_lookup(indi, c[r]).append(['C'])
                    break
            elif i == 'del':  # delete a block randomly
                c = count_list(indi)[1]
                r = random.randint(0, len(c) - 1)
                if not c or len(c[r]) < 2: continue
                del nested_lookup(indi, c[r])[:]
                break
            else:  # replace a block randomly
                c = count_list(indi)[1]
                r = random.randint(0, len(c) - 1)
                while isinstance(nested_lookup(indi, c[r])[0], list):
                    c[r].append(0)
                old_block_code = nested_lookup(indi, c[r])[0]
                for x in self.bb:
                    if x.smiles_struct == old_block_code: old_block = x
                s = random.randint(0, len(self.bb)-1)
                new_block = self.bb[s]
                nested_lookup(indi, c[r])[0] = new_block.smiles_struct
                diff = old_block.spaces - new_block.spaces
                if diff < 0:
                    for _ in range(-diff):
                        nested_lookup(indi, c[r]).append([])
                elif diff > 0:
                    tmp = deepcopy(nested_lookup(indi, c[r])[1:])
                    del nested_lookup(indi, c[r])[1:]
                    nested_lookup(indi, c[r]).append(['C'])
                    for _ in range(new_block.spaces-1): nested_lookup(indi, c[r]).append(random.choice(tmp))
                break
        return tuple(deepcopy(indi))

    def select(self, population, fit_list, num, choice="Roulette"):
        epop, efits = [i[0] for i in fit_list], [i[1] for i in fit_list]
        o_fits = [efits[epop.index(i)] for i in population]

        df_fits = pd.DataFrame(o_fits)
        # calculate distance from cut-offs
        weights = pd.DataFrame([df_fits[i] / self.fit_val[i][1] for i in range(df_fits.shape[1])]).T
        # get weighted objective function values based on max/min
        df_fits = df_fits * (weights.values ** [i[0] for i in self.fit_val])
        # scale all values in range 1-2
        df2 = [((df_fits[i] - df_fits[i].min()) / (df_fits[i].max() - df_fits[i].min())) + 1 for i in range(df_fits.shape[1])]
        # inverse min columns
        df2 = pd.DataFrame([df2[i]**self.fit_val[i][0] for i in range(len(df2))]).T
        # rescale all values in range 1-2
        df2 = pd.DataFrame([((df2[i] - df2[i].min()) / (df2[i].max() - df2[i].min())) + 1 for i in range(df2.shape[1])])

        fitnesses = list(df2.sum())

        if choice == "Roulette":
            total_fitness = float(sum(fitnesses))
            rel_fitness = [f/total_fitness for f in fitnesses]
            # Generate probability intervals for each individual
            probs = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))]
            # Draw new population
            new_population = []
            for _ in range(num):
                r = random.random()
                for i, individual in enumerate(population):
                    if r <= probs[i]:
                        new_population.append(deepcopy(individual))
                        break
            return new_population
        else:
            fits_sort = sorted(fitnesses, reverse=True)
            best = [deepcopy(population[fitnesses.index(fits_sort[i])]) for i in range(min(num, len(population)))]
            return best

    def batch(self, fit_list):
        """Function to run genetic algorithm in batch mode. It writes to file a list of new individuals that are to be evaluated. If running for the first time, fit_list should be an empty tuple.

        Parameters
        ----------
        fit_list: tuple,
            tuple of lists: (GA individual, fitness, rev_smiles)
            formatted with the proper data types for each value in list. Do not send strings.

        """
        if len(fit_list) == 0:
            pop = self.pop_generator(n=self.pop_size)  # list of tuples
            pop_to_write = pop
        else:
            total_pop = [i[0] for i in fit_list]
            # Select the next generation individuals
            pop = self.select(total_pop, fit_list, self.pop_size, choice="best")

            cross_pop, mutant_pop, co_pop = [], [], []
            # Generate crossover population
            co_pop = self.select(pop, fit_list, self.crossover_size)
            shflist = pop + total_pop
            random.shuffle(shflist)
            c_total = co_pop + shflist
            for child1, child2 in zip(c_total[::2], c_total[1::2]):
                if len(cross_pop) >= self.crossover_size: break
                c1, c2 = self.crossover(child1, child2)
                epop = [i[0] for i in fit_list]
                if c1 in epop or c2 in epop or c1 == c2 or c1 in cross_pop or c2 in cross_pop: continue
                cross_pop.extend([deepcopy(c1), deepcopy(c2)])

            # Generate mutation population
            mu_pop = self.select(pop, fit_list, self.mutation_size)
            for mutant in mu_pop + cross_pop + total_pop + pop:
                if len(mutant_pop) >= self.mutation_size: break
                mt = self.custom_mutate(mutant)
                if mt in [i[0] for i in fit_list] or mt in mutant_pop: continue
                mutant_pop.append(mt)
            pop_to_write = cross_pop + mutant_pop

        with open('new_molecules.csv', 'w') as fl:
            fl.write('individual,smiles\n')
            for mols in pop_to_write:
                fl.write(str(mols) + ',' + str(self.list_to_smiles(list(mols))) + '\n')

    def search(self, n_generations=20, init_ratio=0.35, crossover_ratio=0.35):
        """
        Algorithm 1:
            Initial population is instantiated. 
            Roulette wheel selection is used for selecting individuals for crossover and mutation.
            The initial population, crossovered and mutated individuals form the pool of individuals from which the best
            n members are selected as the initial population for the next generation, where n is the size of population.

        Algorithm 2:
            Same as algorithm 1 but when selecting individuals for next generation, n members are selected using Roulette wheel selection.

        Algorithm 3:
            Same as algorithm 1 but when selecting individuals for next generation, best members from each of the three pools (initial population, crossover and mutation) are selected according to the input parameters in the search method.

        Algorithm 4:
            Same as algorithm 1 but mutation population is selected from the crossover population and not from the parents directly.


        Parameters
        ----------
        n_generations: integer, optional (default = 20)
                An integer for the number of generations for evolving the population

        init_ratio: float, optional (default = 0.35)
            Fraction of initial population to select for next generation. Required only for algorithm 3.

        crossover_ratio: float, optional (default = 0.35)
            Fraction of crossover population to select for next generation. Required only for algorithm 3.

        
        Returns
        -------
        best_ind_df:  pandas dataframe
            A pandas dataframe of best individuals of each generation

        best_ind:  dict,
            The best individual after the last generation.

        """
        def fit_eval(invalid_ind, fit_list):
            epop, fit_list = [i[0] for i in fit_list], list(fit_list)
            new_pop = []
            if invalid_ind: 
                invalid_ind = [i for i in invalid_ind if i not in epop]
                obval = [self.pre_eval(i) for i in invalid_ind]
                rev_smiles, fitnesses = [i[0] for i in obval], [i[1] for i in obval]
                for ind, fit, r_smi in zip(invalid_ind, fitnesses, rev_smiles):
                    if fit is not None:
                        fit_list.append((deepcopy(ind), fit, r_smi))
                        new_pop.append(deepcopy(ind))
            return tuple(fit_list), new_pop

        if init_ratio >= 1 or crossover_ratio >= 1 or (init_ratio+crossover_ratio) >= 1:
            raise Exception("Sum of parameters init_ratio and crossover_ratio should be in the range (0,1)")
        if self.population is not None:
            pop = self.population
            fit_list = self.fit_list
        else:
            pop = self.pop_generator(n=self.pop_size)  # list of tuples
            fit_list = ()

        # Evaluate the initial population
        fit_list, pop = fit_eval(pop, fit_list)

        total_pop = []
        for xg in range(n_generations):
            cross_pop, mutant_pop, co_pop, psum = [], [], [], len(fit_list)
            # Generate crossover population
            co_pop = self.select(pop, fit_list, self.crossover_size)
            shflist = pop + total_pop
            random.shuffle(shflist)
            c_total = co_pop + shflist
            for child1, child2 in zip(c_total[::2], c_total[1::2]):
                if (len(fit_list) - psum) >= self.crossover_size: break
                c1, c2 = self.crossover(child1, child2)
                epop = [i[0] for i in fit_list]
                if c1 in epop or c2 in epop or c1 == c2: continue
                fit_list, new_cpop = fit_eval([c1, c2], fit_list)
                cross_pop.extend(new_cpop)

            # Generate mutation population
            if self.algo == 4:
                mu_pop = self.select(cross_pop, fit_list, self.mutation_size)
            else:
                mu_pop = self.select(pop, fit_list, self.mutation_size)

            pre_mu = len(fit_list)
            for mutant in mu_pop + cross_pop + total_pop + pop:
                if (len(fit_list) - pre_mu) >= self.mutation_size: break
                mt = self.custom_mutate(mutant)
                if mt in [i[0] for i in fit_list]: continue
                fit_list, new_mpop = fit_eval([mt], fit_list)
                mutant_pop.extend(new_mpop)

            # Select the next generation individuals
            total_pop = pop + cross_pop + mutant_pop
            if self.algo == 2:
                pop = self.select(total_pop, fit_list, self.pop_size)
            elif self.algo == 3:
                p1 = self.select(pop, fit_list, int(init_ratio*self.pop_size), choice="best")
                p2 = self.select(cross_pop, fit_list, int(crossover_ratio*self.pop_size), choice="best")
                p3 = self.select(mutant_pop, fit_list, self.pop_size-len(p1)-len(p2), choice="best")
                pop = p1 + p2 + p3
            else:
                pop = self.select(total_pop, fit_list, self.pop_size, choice="best")

            # Saving each generation
            if xg == 0:
                df_gen = pd.DataFrame([i for i in fit_list])
            else:
                df_gen = pd.DataFrame([i for i in fit_list[-(self.crossover_size+self.mutation_size):]])
            df_gen = df_gen[[2, 1]]
            df_gen.columns = ['Canonical SMILES', 'Fitness Values']
            fname = '/generation_' + str(xg+1) + '.csv'
            df_gen.to_csv(os.path.join(self.output_dir + fname), index=None)

        self.population = pop  # stores best individuals of last generation
        self.fit_list = fit_list

def count_list(l):
    """ Function to get the nested indices of empty and filled lists generated by genetic algorithm class.

    Parameters
    ----------
    l: list,
        nested list of list

    Returns
    -------
    e_list_index: list,
        the nested indices of empty lists
    f_list_index: list,
        the nested indices of filled/non-empty lists

    """
    e_list_index, f_list_index, iterate_over = [], [], []
    for e, g in zip(l, range(len(l))):
        if isinstance(e, list):
            if not e:
                temp = [g]
                e_list_index.append(temp)
            else:
                if e != ['C']:
                    temp = [g]
                    f_list_index.append(temp)
                    iterate_over.append(e)

    while len(iterate_over) != 0:
        f_list = []
        for x, prefactor in zip(iterate_over, f_list_index[-len(iterate_over):]):
            if not f_list_index:
                prefactor = []
            for e, g in zip(x, range(len(x))):
                if isinstance(e, list):
                    if e == x:  # this is to check for self-referential lists
                        e_list_index.append(prefactor + temp)
                    else:
                        if not e:
                            temp = [g]
                            e_list_index.append(prefactor + temp)
                        else:
                            if e != ['C']:
                                temp = [g]
                                f_list_index.append(prefactor + temp)
                                f_list.append(e)

        del iterate_over[:]
        for items in f_list:
            iterate_over.append(items)
    return e_list_index, f_list_index

def nested_lookup(n, idexs):
    """Function to fetch a nested sublist given its nested indices.

    Parameters
    ----------
    n: list,
        the main list in which to look for the sublist

    idexs: list,
        the indices of the sublist 

    Returns
    -------
    list: sublist with given indices

    """
    if len(idexs) == 1:
        return n[idexs[0]]
    return nested_lookup(n[idexs[0]], idexs[1:])








































































































