import sys
from openbabel import pybel
from openbabel.openbabel import OBAtomAtomIter
import scipy
from collections import defaultdict
import os
from itertools import chain
import pandas as pd
import time
import random
import math
from copy import deepcopy

def log_error(sentence, output_dir, mpidict, msg="Aborting the run", error=False):
    """Print to both error file and logfile and then exit code.
    Parameters: sentence (str), output_dir (str), mpidict (dict), msg (str), error (bool)
    Output: None
    """
    rank = mpidict['rank']
    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logfile = open(os.path.join(output_dir+'/logfile.txt'),'a')
        print(sentence)
        logfile.write(str(sentence) + "\n")
    if error:
        if rank == 0:
            error_file = open(os.path.join(output_dir+'/error_file.txt'),'a')
            error_file.write(str(sentence) + "\n")
            sys.exit(msg)
        else:
            sys.exit()

def check_building_blocks(smiles, line, file_name, output_dir, mpidict):
    """Validate the building blocks input (smiles or inchi) and return the smiles of the molecule.
    Parameters: smiles (str), line (int), file_name (str), output_dir (str), mpidict (dict)
    Output: smiles (str)
    """
    inchi_bb, smiles_bb = True, True
    try:
        mol = pybel.readstring("inchi",smiles)
        smiles = str(mol).strip()
    except:
        inchi_bb = False
    
    try:
        mol = pybel.readstring("smi",smiles)
    except:
        smiles_bb = False

    if not inchi_bb and not smiles_bb:
        tmp_str = f"Error: The SMILES/InChI string('{smiles}') provided in line {line} of data file '{file_name}' is not valid. Please provide correct SMILES/InChI."
        log_error(tmp_str, output_dir, mpidict, "Aborting due to wrong molecule description.", error=True)
    else:
        return smiles

    return smiles

def molecule(smiles, code):
    """Create a dictionary for each molecule.
    Parameters: smiles (str), code (str)
    Output: mol (dict)
    """
    mol = {'smiles': smiles, 'code': code}
    obm = pybel.readstring("smi", smiles)
    mol['can_smiles'] = obm.write("can")
    obm_can = pybel.readstring("smi", mol['can_smiles'])
    mol['reverse_smiles'] = reverse_mol(obm_can, list(obm_can.atoms))
    return mol

class OutputGrabber(object):
    """Class used to grab standard output/another stream/system errors from any program.
    Parameters: stream (stream), threaded (bool)
    Output: None
    """
    def __init__(self, stream=None, threaded=False):
        self.origstream = stream
        self.origstream3 = stream
        self.origstream2 = sys.stdout
        self.origstreamfd = self.origstream.fileno()
        self.capturedtext = ""
        self.pipe_out, self.pipe_in = os.pipe()

    def start(self):
        """Start capturing the stream data.
        Parameters: None
        Output: None
        """
        self.capturedtext = ""
        self.streamfd = os.dup(self.origstreamfd)
        os.dup2(self.pipe_in, self.origstreamfd)

    def stop(self):
        """Stop capturing the stream data and save the text in `capturedtext`.
        Parameters: None
        Output: None
        """
        os.dup2(self.streamfd, self.origstreamfd)
        os.close(self.pipe_out)

out = OutputGrabber(sys.stderr)

def reverse_mol(mol, atoms):
    """Convert a molecule's potential reactive sites into H atoms.
    Parameters: mol (object), atoms (list)
    Output: smiles (str)
    """
    atom_num = []
    myFr = pybel.readstring('smi', "[Fr]")
    Fratom = myFr.OBMol.GetAtom(1)
    for atom in atoms:
        atom_num.append(atom.OBAtom.GetAtomicNum())
    
    if 88 in atom_num:
        for atom in atoms:
            hcount = atom.OBAtom.ExplicitHydrogenCount() + atom.OBAtom.GetImplicitHCount()
            index = atom.OBAtom.GetIdx()
            while hcount != 0:
                size = len(list(mol.atoms))
                mol.OBMol.InsertAtom(Fratom)
                mol.OBMol.AddBond(index, size+1, 1, 0, -1)
                x = mol.OBMol.GetAtom(index).GetImplicitHCount() - 1
                if x >= 0: mol.OBMol.GetAtom(index).SetImplicitHCount(x)
                hcount = atom.OBAtom.ExplicitHydrogenCount() + atom.OBAtom.GetImplicitHCount()
                
        atoms = list(mol.atoms)
        for atom in atoms:
            if atom.OBAtom.GetAtomicNum() == 88:
                atom.OBAtom.SetAtomicNum(1)
                   
    smiles = mol.write("can")[:-2]
    return smiles

def lipinski(mol):
    """Return the values of the Lipinski descriptors.
    Parameters: mol (object)
    Output: desc (dict)
    """
    HBD = pybel.Smarts("[#7,#8;!H0]")
    HBA = pybel.Smarts("[#7,#8]")

    desc = {
        'molwt': mol.molwt,
        'HBD': len(HBD.findall(mol)),
        'HBA': len(HBA.findall(mol)),
        'logP': mol.calcdesc(['logP'])['logP']
    }
    return desc

def unique_structs(mol, smarts):
    """Calculate the number of given sub-structures (SMARTS) in molecule provided.
    Parameters: mol (object), smarts (str)
    Output: num_unique_matches (int)
    """
    smarts = pybel.Smarts(smarts)
    smarts.obsmarts.Match(mol.OBMol)
    num_unique_matches = len(smarts.findall(mol))
    return num_unique_matches

def get_rules(config_file, output_dir, mpidict):
    """Read generation rules provided in the config file.
    Parameters: config_file (file), output_dir (str), mpidict (dict)
    Output: rules_dict (dict), lib_args (list)
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
                    tmp_str = "ERROR: Wrong generation rule provided for "+line
                    log_error(tmp_str, output_dir, mpidict, "Aborting due to wrong generation rule.", error=True)
                rules_dict['include_bb'] = [i.strip() for i in eval(value)]
                continue
            elif i == 11:
                if isinstance(eval(value), tuple):
                    rules_dict['heteroatoms'] = eval(value)
                else:
                    tmp_str = "ERROR: Wrong generation rule provided for "+line
                    log_error(tmp_str, output_dir, mpidict, "Aborting due to wrong generation rule.", error=True)
                continue
            elif i == 12:
                if value == "True":
                    rules_dict['lipinski'] = True
                elif value == "False":
                    continue
                else:
                    tmp_str = "ERROR: Wrong generation rule provided for "+line
                    log_error(tmp_str, output_dir, mpidict, "Aborting due to wrong generation rule.", error=True)
                continue
            elif i == 13:
                target_mols = value.split(',')
                smiles_to_comp = []
                for j in target_mols:
                    target_smiles, tanimoto_index = j.split('-')[0].strip(), j.split('-')[1].strip()
                    smiles = check_building_blocks(target_smiles, i+1, config_file, output_dir, mpidict)
                    smiles_to_comp.append([smiles, tanimoto_index])
                rules_dict['fingerprint'] = smiles_to_comp
                continue
            elif i == 14 or i == 15:
                smiles_l = []
                for item in value.split(','):
                    smiles = check_building_blocks(item.strip(), i+1, config_file, output_dir, mpidict)
                    smiles_l.append(smiles)
                rules_dict[str(i)] = smiles_l
                continue
            elif i == 16:
                if value != 'True' and value != 'False':
                    tmp_str = "ERROR: Wrong generation rule provided for "+line
                    tmp_str = tmp_str+"Provide either True or False. \n"
                    log_error(tmp_str, output_dir, mpidict, "Aborting due to wrong generation rule.", error=True)
                if value == 'False':
                    rules_dict['bb_final_lib'] = False
                elif value == 'True':
                    rules_dict['bb_final_lib'] = True
                continue
            elif value != 'None':
                if not isinstance(eval(value), tuple) or len(eval(value)) != 2:
                    tmp_str = "ERROR: Wrong generation rule provided for "+line
                    tmp_str = tmp_str+"Provide the range in tuple format (min, max). \n"
                    log_error(tmp_str, output_dir, mpidict, "Aborting due to wrong generation rule.", error=True)
                else:
                    rules_dict[str(i)] = eval(value)
        elif '::' in line:
            words = line.split('::')
            value = words[1].strip()
            lib_args.append(value)
    return rules_dict, lib_args

class GeneticAlgorithm:
    """A genetic algorithm class for search or optimization problems.
    Parameters: evaluate (function), fitness (tuple), crossover_size (int), mutation_size (int), algorithm (int), initial_mols (list), rules_dict (dict), output_dir (str), mpidict (dict)
    Output: None
    """
    def __init__(self, evaluate, fitness, crossover_size, mutation_size, algorithm, initial_mols, rules_dict, output_dir, mpidict):
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
        self.bb = [building_blocks(i) for i in initial_mols]
        self.rules_dict = rules_dict

    def pop_generator(self, n):
        """Generate the initial population.
        Parameters: n (int)
        Output: pop (list)
        """
        pop = []
        for _ in range(n):
            pop.append(tuple(self.chromosome_generator()))
        return pop

    def chromosome_generator(self):
        """Generate the chromosome for the algorithm.
        Parameters: None
        Output: chromosome (list)
        """   
        i = 0
        chromosome = []
        ind_len = random.randint(2, 5)
        while i < ind_len:
            if i == 0:
                r = random.randint(0, len(self.bb) - 1)
                chromosome.append(self.bb[r].smiles_struct)
                for j in range(self.bb[r].spaces):
                    chromosome.append([])
            else:
                avl_pos = count_list(chromosome)[0]
                if len(avl_pos) <= 0:
                    return chromosome
                r = random.randint(0, len(avl_pos) - 1)
                s = random.randint(0, len(self.bb) - 1)
                t = random.randint(1, self.bb[s].spaces)
                nested_lookup(chromosome, avl_pos[r]).append(self.bb[s].smiles_struct)
                for j in range(self.bb[s].spaces):
                    if (j+1) != t:
                        nested_lookup(chromosome, avl_pos[r]).append([])
                    else:
                        nested_lookup(chromosome, avl_pos[r]).append(['C'])
            i += 1
        return deepcopy(chromosome)

def list_to_smiles(self, indi_list):
    """Convert the lists of lists generated by the algorithm to actual molecules.
    Parameters: indi_list (list)
    Output: mol_combi (str)
    """
    mol_code_list, parent_list, handle_list, mol_combi, mol_len = [], [], [], '', 0
    f_lists = count_list(indi_list)[1]
    parent_list.append([indi_list, indi_list, -100])
    while len(parent_list) != 0:
        iterate_over = parent_list[-1][1]
        new_item = False
        for k_ind, k in enumerate(iterate_over):
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
                        handle_list.append([handle_1, handle_2])
                    else:
                        parent_list[-1][2] = k_ind
                        parent_list.append([iterate_over, k, -100])
                        new_item = True
                        break
        if not new_item:
            del parent_list[-1]
    mol_combi = pybel.readstring('smi', mol_combi)
    for handles in handle_list:
        mol_combi.OBMol.AddBond(handles[0], handles[1], 1)
        x = mol_combi.OBMol.GetAtom(handles[0]).GetImplicitHCount() - 1
        y = mol_combi.OBMol.GetAtom(handles[1]).GetImplicitHCount() - 1
        if x >= 0: mol_combi.OBMol.GetAtom(handles[0]).SetImplicitHCount(x)
        if y >= 0: mol_combi.OBMol.GetAtom(handles[1]).SetImplicitHCount(y)
    for atom in list(mol_combi.atoms):
        if atom.OBAtom.GetAtomicNum() == 87 or atom.OBAtom.GetAtomicNum() == 88:
            atom.OBAtom.SetAtomicNum(1)
    mol_combi = mol_combi.write("can")
    return mol_combi

def pre_eval(self, indi_list):
    """Pre-process the individuals/chromosomes before sending them to the evaluate function.
    Parameters: indi_list (list)
    Output: fit_val (float)
    """
    mol_combi_smiles = self.list_to_smiles(deepcopy(list(indi_list)))
    mol_combi = pybel.readstring("smi", mol_combi_smiles)
    if not if_add(mol_combi_smiles, self.rules_dict, code='a'):
        return mol_combi_smiles, None
    else:
        fit_val = self.evaluate(mol_combi)
        if isinstance(fit_val, (tuple, list)):
            return mol_combi_smiles, tuple(fit_val)
        else:
            return mol_combi_smiles, tuple([fit_val])

def crossover(self, child1, child2):
    """Perform crossover on two individuals.
    Parameters: child1 (list), child2 (list)
    Output: child1 (list), child2 (list)
    """
    child1, child2 = deepcopy(list(child1)), deepcopy(list(child2))
    c1 = count_list(child1)[1]
    c2 = count_list(child2)[1]
    if not (len(c1)==0 or len(c2)==0):
        r1 = random.randint(0, len(c1) - 1)
        r2 = random.randint(0, len(c2) - 1)
        t1 = nested_lookup(child1, c1[r1])
        t2 = nested_lookup(child2, c2[r2])
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
    """Perform mutation on an individual.
    Parameters: indi (list)
    Output: indi (list)
    """
    indi = deepcopy(list(indi))
    t = ['add', 'del', 'replace']
    random.shuffle(t)
    for i in t:
        if i == 'add':
            c = count_list(indi)[0]
            if not c:
                continue
            else:
                r = random.randint(0, len(c) - 1)
                s = random.randint(0, len(self.bb)-1)
                t = random.randint(1, self.bb[s].spaces)
                nested_lookup(indi, c[r]).append(self.bb[s].smiles_struct)
                for j in range(self.bb[s].spaces):
                    if (j+1) != t:
                        nested_lookup(indi, c[r]).append([])
                    else:
                        nested_lookup(indi, c[r]).append(['C'])
                break
        elif i == 'del':
            c = count_list(indi)[1]
            r = random.randint(0, len(c) - 1)
            if not c or len(c[r]) < 2: continue
            del nested_lookup(indi, c[r])[:]
            break
        else:
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
                for i in range(new_block.spaces-1): nested_lookup(indi, c[r]).append(random.choice(tmp))
            break
    return tuple(deepcopy(indi))

def select(self, population, fit_list, num, choice="Roulette"):
    """Select individuals from the population.
    Parameters: population (list), fit_list (list), num (int), choice (str)
    Output: new_population (list)
    """
    epop, efits = [i[0] for i in fit_list], [i[1] for i in fit_list]
    o_fits = [efits[epop.index(i)] for i in population]
    df_fits = pd.DataFrame(o_fits)
    weights = pd.DataFrame([df_fits[i] / self.fit_val[i][1] for i in range(df_fits.shape[1])]).T
    df_fits = df_fits * (weights.values ** [i[0] for i in self.fit_val])
    df2 = [((df_fits[i] - df_fits[i].min()) / (df_fits[i].max() - df_fits[i].min())) + 1 for i in range(df_fits.shape[1])]
    df2 = pd.DataFrame([df2[i]**self.fit_val[i][0] for i in range(len(df2))]).T
    df2 = pd.DataFrame([((df2[i] - df2[i].min()) / (df2[i].max() - df2[i].min())) + 1 for i in range(df2.shape[1])])
    fitnesses = list(df2.sum())
    if choice == "Roulette":
        total_fitness = float(sum(fitnesses))
        rel_fitness = [f/total_fitness for f in fitnesses]
        probs = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))]
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
    """Run genetic algorithm in batch mode.
    Parameters: fit_list (list)
    Output: None
    """
    if len(fit_list) == 0:
        pop = self.pop_generator(n=self.pop_size)
        pop_to_write = pop
    else:
        total_pop = [i[0] for i in fit_list]
        pop = self.select(total_pop, fit_list, self.pop_size, choice="best")
        cross_pop, mutant_pop, co_pop = [], [], []
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
    """Run genetic algorithm search.
    Parameters: n_generations (int), init_ratio (float), crossover_ratio (float)
    Output: best_ind_df (pd.DataFrame), best_ind (dict)
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

    if init_ratio >=1 or crossover_ratio >=1 or (init_ratio+crossover_ratio)>=1: raise Exception("Sum of parameters init_ratio and crossover_ratio should be in the range (0,1)")
    if self.population is not None:
        pop = self.population
        fit_list = self.fit_list
    else:
        pop = self.pop_generator(n=self.pop_size)
        fit_list = ()
    fit_list, pop = fit_eval(pop, fit_list)
    total_pop = []
    for xg in range(n_generations):
        cross_pop, mutant_pop, co_pop, psum = [], [], [], len(fit_list)
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
        if xg == 0:
            df_gen = pd.DataFrame([i for i in fit_list])
        else:
            df_gen = pd.DataFrame([i for i in fit_list[-(self.crossover_size+self.mutation_size):]])
        df_gen = df_gen[[2, 1]]
        df_gen.columns = ['Canonical SMILES', 'Fitness Values']
        fname = '/generation_' + str(xg+1) + '.csv'
        df_gen.to_csv(os.path.join(self.output_dir + fname), index=None)
    self.population = pop
    self.fit_list = fit_list
    best_ind_df = df_gen
    best_ind = pop[0]
    return best_ind_df, best_ind

def count_list(l):
    """Get the nested indices of empty and filled lists generated by genetic algorithm class.
    Parameters: l (list)
    Output: e_list_index (list), f_list_index (list)
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
                    if e == x:
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
    """Fetch a nested sublist given its nested indices.
    Parameters: n (list), idexs (list)
    Output: list (sublist)
    """
    if len(idexs) == 1:
        return n[idexs[0]]
    return nested_lookup(n[idexs[0]], idexs[1:])

class BuildingBlocks:
    """A class for each of the building blocks that are read from the building blocks file.
    Parameters: mol (dict)
    Output: None
    """
    def __init__(self, mol):
        self.smiles = mol['reverse_smiles']
        self.smiles_struct = mol['code']
        mol_ob = pybel.readstring("smi", mol['reverse_smiles'])
        self.atom_len = len(list(mol_ob.atoms))
        self.index_list = get_index_list(mol, list(mol_ob.atoms))
        self.spaces = len(self.index_list)

def library_generator(config_file='config.dat', building_blocks_file='building_blocks.dat', output_dir='./', genetic_algorithm_config=None, cost_function=None, fitnesses_list=None):
    """Main wrapper function for library generation.
    Parameters: config_file (str), building_blocks_file (str), output_dir (str), genetic_algorithm_config (str), cost_function (function), fitnesses_list (list)
    Output: None
    """
    try:
        from mpi4py import MPI
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
        log_error("Warning: MPI4PY not found. Chemgenpy not running in parallel.\n\n", output_dir, mpidict)
    
    txt = "\n\n\n============================================================================================================"
    txt += "\n Chemgenpy - A Molecular Library Generator that can generate large libraries"
    txt += "\n============================================================================================================\n\n\n"
    log_error(txt, output_dir, mpidict)
    log_error("===================================================================================================", output_dir, mpidict)
    
    try:
        rulesFile = open(config_file)
    except:
        tmp_str = "Config file does not exist. Please provide correct config file.\n"
        log_error(tmp_str, output_dir, mpidict, "Aborting due to wrong file.", error=True)
    log_error("Reading generation rules", output_dir, mpidict)
    log_error("===================================================================================================", output_dir, mpidict)
    rules_dict, args = get_rules(rulesFile, output_dir, mpidict)
    BB_file = building_blocks_file
    combi_type, gen_len, outfile_type, max_fpf, lib_name = args
    gen_len, max_fpf = int(gen_len), int(max_fpf)
    if gen_len == 0:
        rules_dict['bb_final_lib'] = True

    initial_mols = []
    log_error("===================================================================================================", output_dir, mpidict)
    log_error(f"Reading building blocks from the file {BB_file}", output_dir, mpidict)
    log_error("===================================================================================================", output_dir, mpidict)
    try:
        infile = open(BB_file)
    except:
        tmp_str = f"Building blocks file {BB_file} does not exist. Please provide correct building blocks file.\n"
        log_error(tmp_str, output_dir, mpidict, "Aborting due to wrong file.", error=True)
    i_smi_list = []
    for i, line in enumerate(infile):
        smiles = line.strip()
        if smiles.isspace() or len(smiles) == 0 or smiles[0] == '#':
            continue
        if '[X]' in smiles:
            smiles = smiles.replace('[X]', '[Ra]')
        smiles = check_building_blocks(smiles, i + 1, BB_file, output_dir, mpidict)
        temp = molecule(smiles, 'F' + str(len(initial_mols) + 1))
        is_duplicate = False
        for z in initial_mols:
            if temp['can_smiles'] not in z['can_smiles']:
                continue
            is_duplicate = True
        if not is_duplicate:
            initial_mols.append(temp)
            i_smi_list.append(temp['can_smiles'])

    log_error(f'Number of building blocks provided = {len(initial_mols)}\n', output_dir, mpidict)
    log_error('unique SMILES: ', output_dir, mpidict)
    log_error(i_smi_list, output_dir, mpidict)

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
            log_error(tmp_str, output_dir, mpidict, "Aborting due to wrong file.", error=True)
        log_error("Reading parameters for genetic algorithm", output_dir, mpidict)
        log_error("===================================================================================================", output_dir, mpidict)
        batch, fitness, crossover_size, mutation_size, algorithm, generations, init_ratio, crossover_ratio = parse_ga(ga_config, output_dir, mpidict)
        
        ga_libgen = GeneticAlgorithm(evaluate=cost_function, fitness=fitness, crossover_size=crossover_size, mutation_size=mutation_size, algorithm=algorithm, initial_mols=initial_mols, rules_dict=rules_dict, output_dir=output_dir, mpidict=mpidict)

        if not batch:
            if cost_function is None:
                log_error("Missing input for genetic algorithm. Provide cost function. Aborting", output_dir, mpidict, error=True)
            ga_libgen.search(n_generations=generations, init_ratio=init_ratio, crossover_ratio=crossover_ratio)
        else:
            if fitnesses_list is None:
                log_error("Missing input for genetic algorithm. Provide fitnesses_list. Aborting", output_dir, mpidict, error=True)
            ga_libgen.batch(fitnesses_list)
        
        return ga_libgen

    log_error('\n\n\n\n\n===================================================================================================', output_dir, mpidict)
    log_error('Generating molecules', output_dir, mpidict)
    log_error('===================================================================================================\n', output_dir, mpidict)
    final_list = generator(init_mol_list=initial_mols, combi_type=combi_type, gen_len=gen_len, rules_dict=rules_dict, output_dir=output_dir, mpidict=mpidict)
    log_error(f'Total number of molecules generated = {len(final_list)}\n\n\n\n\n', output_dir, mpidict)
    log_error("===================================================================================================\n\n\n", output_dir, mpidict)

    df_final_list = pd.DataFrame(final_list)
    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df_final_list.drop(['smiles', 'can_smiles'], axis=1).to_csv(os.path.join(output_dir+'final_library.csv'), index=None)

    if outfile_type == 'smi':
        if rank == 0:
            if not os.path.exists(output_dir + lib_name + '_' + outfile_type):
                os.makedirs(output_dir + lib_name + '_' + outfile_type)
            outdata = output_dir + lib_name + '_' + outfile_type + "/final_smiles.csv"
            log_error(f'Writing SMILES to file \'{outdata}\'\n', output_dir, mpidict)
            df_new = df_final_list['reverse_smiles'].copy()
            df_new.to_csv(outdata, index=False, header=False)
    else:
        log_error(f'Writing molecules with molecule type {outfile_type}\n', output_dir, mpidict)
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
            mol_ob = pybel.readstring("smi", smiles_list[i]['reverse_smiles'])
            mymol = pybel.readstring("smi", mol_ob.write("can"))
            mymol.make3D(forcefield='mmff94', steps=50)
            mymol.write(outfile_type, output_dir + lib_name + '_' + outfile_type + "/" + str(folder_no) + "_" + str(max_fpf) + "/" + str(val + 1) + "." + outfile_type, overwrite=True)
            if (val + 1) % max_fpf == 0:
                folder_no = folder_no + 1
        
    log_error('File writing terminated successfully.\n', output_dir, mpidict)
    if comm is not None:
        wt2 = MPI.Wtime()
    else:
        wt2 = time.time()
    log_error(f'Total time_taken: {str("%.3g" % (wt2 - wt1))}\n', output_dir, mpidict)
    return None



