import os
import pybel
from IPython.display import display, clear_output, Javascript
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
import ipywidgets as widgets
from ipywidgets import Layout, Label, interact

# Global Variables
style = {'description_width': 'initial', 'font_weight': 'bold'}
BB_list = []
smiles = []

# Run this function to start the GUI builder.
def config_builder():
    display(widgets.HTML(value="<font color=green><font size=5><b><u>CHEMGENPY</font>"))
    input_direc_intro = widgets.HTML(description="Provide the directory to save all the input files", style=style)
    input_direc = widgets.Text(description="Input file directory", placeholder="", style=style)
    input_direc_button = widgets.Button(description='Continue', layout=Layout(width='20%', border='solid 1px black'), style=style)
    
    inp = widgets.VBox(children=[input_direc_intro, input_direc, input_direc_button])
    display(inp)

    def on_input_clicked(o):
        display(widgets.HTML(value="<font color=green><font size=5><b><u>BUILDING BLOCKS</font>"))
        building_blocks(input_direc.value)
    
    input_direc_button.on_click(on_input_clicked)

def building_blocks(input_direc):
    space_box = widgets.Box(layout=widgets.Layout(height='20px', width='90%')) 

    # Section for providing SMILES input
    name = widgets.Text(description="File name", placeholder="Type the file to be imported", style=style)
    BB = widgets.Text(description='Building Blocks', style=style)  # Textbox for SMILES input
    type_smiles_intro = widgets.HTML("""<font size=3>Create a building blocks file by entering individual/comma separated SMILES of each building block.""",
                                     layout=widgets.Layout(height='55px', width='90%', size='20'))
    another = widgets.Button(description='Add building block', layout=Layout(width='45%', border='solid 1px black'), style=style)  # Button for adding another building block

    # Section for uploading building blocks file
    existing_file_intro = widgets.HTML("""<font size=3>Upload a file containing all the building blocks""",
                                       layout=widgets.Layout(height='60px', width='90%', size='20'))
    enter = widgets.Button(description='Upload file', layout=Layout(width='30%', border='solid 1px black'), style=style)
    existing_filebox = widgets.VBox(children=[existing_file_intro, space_box, name, space_box, enter])

    def upload(e):
        file_name = name.value
        if file_name:
            if os.path.exists(file_name):
                with open(file_name, "r") as existing:
                    lines = existing.readlines()
                    if not lines:
                        print("Building blocks file is empty")
                    else:
                        process_lines(lines)
            else:
                print("The building blocks file does not exist")
    
    def process_lines(lines):
        for line in lines:
            if line.startswith(('#', '\n', '\t')):
                continue
            else:
                try:
                    mol = pybel.readstring("smi", line)
                    smiles.append(line)
                except:
                    print(f"Incorrect SMILES: {line}")

        if smiles:
            visualize_smiles(smiles)
            generation_rules()

    def visualize_smiles(smiles_list):
        mol_lists = [Chem.MolFromSmiles(smile) for smile in smiles_list]
        [mol.SetProp('_Name', 'B' + str(i)) for i, mol in enumerate(mol_lists)]
        ibu1 = Chem.Draw.MolsToGridImage(mol_lists)
        display(ibu1)

    enter.on_click(upload)

    def on_another_clicked(i):
        clear_output()
        process_individual_smiles(BB.value)
        display(widgets.HTML(value="<font color=green><font size=5><b><u>BUILDING BLOCKS</font>"))
        display(tab1)

    def process_individual_smiles(smiles_input):
        global BB_list
        BB_list.extend(smiles_input.split(','))

    final_BB = widgets.Button(description='Create building blocks file', layout=Layout(width='45%', border='solid 1px black'), style=style)

    def on_button_clicked(d):
        create_building_blocks_file(input_direc, BB_list)

    def create_building_blocks_file(input_direc, BB_list):
        file_path = os.path.join(input_direc, "building_blocks.dat") if input_direc else "building_blocks.dat"
        with open(file_path, "w") as building_blocks:
            for block in BB_list:
                building_blocks.write(block + '\n')
        generation_rules()

    def on_visualization_clicked(t):
        try:
            visualize_smiles(BB.value.split(','))
        except:
            print("You typed a wrong SMILES. Unable to visualize")

    visualize = widgets.Button(description='Visualize', layout=Layout(width='45%', border='solid 1px black'), style=style)
    visualize.on_click(on_visualization_clicked)
    final_BB.on_click(on_button_clicked)
    another.on_click(on_another_clicked)
    type_smilesbox = widgets.VBox(children=[type_smiles_intro, BB, space_box, visualize, another, final_BB])
    tab1 = widgets.Tab(children=[type_smilesbox, existing_filebox], style=style)
    tab1.set_title(0, 'Individual SMILES')
    tab1.set_title(1, 'Upload file')
    display(tab1)

def generation_rules():
    space_box = widgets.Box(layout=widgets.Layout(height='55px', width='90%'))
    second = widgets.Button(description='Next section', layout=Layout(width='auto', border='solid 1px black'), style=style)
    display(second)

    def second_section(q):
        if os.path.exists("building_blocks.dat") and os.stat("building_blocks.dat").st_size > 0:
            display(widgets.HTML(value="<font color=green><font size=5><b><u>USER DEFINED CONSTRAINTS</font>"))
            build_constraints_form()

    second.on_click(second_section)

def build_constraints_form():
    # UI components for user-defined constraints
    building_blocks = widgets.Text(value='None', placeholder='Enter building blocks', description='Specific Building Blocks:', style=style)
    bb_intro = widgets.HTML("""Specify the building blocks which must be present in all the molecules in the final library""", layout=widgets.Layout(height='45px', width='90%', size='20'))
    bb = widgets.VBox(children=[bb_intro, widgets.Box(layout=widgets.Layout(height='20px')), building_blocks])
    
    constraints_widgets = create_constraints_widgets()
    accordion = create_constraints_accordion(constraints_widgets)
    
    display(accordion)
    submit = widgets.Button(description="Submit", layout=Layout(width='20%', border='solid 1px black'), style=style)
    display(submit)
    submit.on_click(lambda x: command_line_arguments(building_blocks, constraints_widgets))

def create_constraints_widgets():
    # Creating widgets for each type of constraint
    style = {'description_width': 'initial'}
    widgets_list = {
        'bonds': create_min_max_widget('Bonds'),
        'atoms': create_min_max_widget('Atoms'),
        'molecular_weight': create_min_max_widget('Molecular Weight'),
        'rings': create_min_max_widget('Rings'),
        'aromatic_rings': create_min_max_widget('Aromatic Rings'),
        'nonaromatic_rings': create_min_max_widget('Non Aromatic Rings'),
        'single_bonds': create_min_max_widget('Single Bonds'),
        'double_bonds': create_min_max_widget('Double Bonds'),
        'triple_bonds': create_min_max_widget('Triple Bonds'),
        'specific_atoms': widgets.Text(value='None', description='Element:', style=style),
        'lipinski_rule': widgets.RadioButtons(options=['True', 'False'], value='False', description='Lipinski rule:', style=style),
        'fingerprint_matching': widgets.Text(value='None', placeholder='Finger print match', description='Finger print matching:', style=style),
        'substructure': widgets.Text(value='None', placeholder='Type the substructure SMILES', description='Substructure:', style=style),
        'substructure_exclusion': widgets.Text(value='None', placeholder='Type the substructure exclusion SMILES', description='Substructure exclusion:', style=style),
        'include_BB': widgets.RadioButtons(options=['True', 'False'], value='True', description='Include BB:', style=style)
    }
    return widgets_list

def create_min_max_widget(description):
    style = {'description_width': 'initial'}
    return widgets.HBox([widgets.Text(description='', value='None', layout=Layout(width='30%'), style=style), widgets.Text(description='', value='None', layout=Layout(width='30%'), style=style)])

def create_constraints_accordion(widgets_list):
    accordion_titles = ['Include Building Blocks', 'Min max', 'Heteroatoms', 'Lipinski Rule', 'Fingerprint Matching', 'Substructure Inclusion', 'Substructure Exclusion', 'Include initial Building Blocks']
    accordion_widgets = [
        widgets.VBox(children=[widgets_list['specific_atoms']]),
        widgets.VBox(children=[create_min_max_description(), widgets_list['bonds'], widgets_list['atoms'], widgets_list['molecular_weight'], widgets_list['rings'], widgets_list['aromatic_rings'], widgets_list['nonaromatic_rings'], widgets_list['single_bonds'], widgets_list['double_bonds'], widgets_list['triple_bonds']]),
        widgets.VBox(children=[widgets_list['specific_atoms']]),
        widgets.VBox(children=[widgets_list['lipinski_rule']]),
        widgets.VBox(children=[widgets_list['fingerprint_matching']]),
        widgets.VBox(children=[widgets_list['substructure']]),
        widgets.VBox(children=[widgets_list['substructure_exclusion']]),
        widgets.VBox(children=[widgets_list['include_BB']])
    ]
    accordion = widgets.Tab(children=accordion_widgets, style=style)
    for i, title in enumerate(accordion_titles):
        accordion.set_title(i, title)
    return accordion

def create_min_max_description():
    return widgets.HTML("""Specify the minimum and maximum values of the following:""", layout=widgets.Layout(height='45px', width='90%', size='20'))

def command_line_arguments(building_blocks, constraints_widgets):
    display(widgets.HTML(value="<font color=green><font size=5><b><u>COMMAND LINE ARGUMENTS</font>"))
    combination_type = widgets.RadioButtons(options=['Fusion', 'Link'], value='Link', description='Combination type:', style=style)
    generation_level = widgets.BoundedIntText(value=1, min=1, max=100, step=1, description='Generation level:', style=style)
    output_type = widgets.Dropdown(options=['smi', 'xyz'], value='smi', description='Output Type:', style=style)
    max_files = widgets.IntText(value=10000, description='Maximum files per folder:', style=style)
    library_name = widgets.Text(value='new_library_', placeholder='Type the name of the library', description='Library Name:', style=style)
    directory = widgets.Text(value="", placeholder='Enter path', description='Directory path', style=style)

    arguments_widgets = [combination_type, generation_level, output_type, max_files, library_name, directory]
    arguments_titles = ['Combination Type', 'No of generations', 'Output File Format', 'Maximum No of Files', 'Library Name', 'Output Directory']

    arguments_tab = widgets.Tab(children=[widgets.VBox([widgets.HTML(f"Specify the {title}"), widget]) for title, widget in zip(arguments_titles, arguments_widgets)])
    for i, title in enumerate(arguments_titles):
        arguments_tab.set_title(i, title)
    display(arguments_tab)

    submit_button = widgets.Button(description="Generate configuration file", layout=Layout(width='20%', border='solid 1px black'), style=style)
    submit_button.on_click(lambda b: generate_configuration_file(building_blocks, constraints_widgets, combination_type, generation_level, output_type, max_files, library_name, directory))
    display(submit_button)

def generate_configuration_file(building_blocks, constraints_widgets, combination_type, generation_level, output_type, max_files, library_name, directory):
    config_file_path = "config.dat"
    if directory.value:
        config_file_path = os.path.join(directory.value, config_file_path)

    with open(config_file_path, "w") as config_file:
        config_file.write("Please input generation rules below. Do not change the order of the options\n")
        config_file.write(f"1. Include building blocks == {building_blocks.value}\n")
        write_min_max_constraints(config_file, constraints_widgets)
        config_file.write(f"11. Max no. of specific atoms == {constraints_widgets['specific_atoms'].value}\n")
        config_file.write(f"12. Lipinski's rule == {constraints_widgets['lipinski_rule'].value}\n")
        config_file.write(f"13. Fingerprint matching == {constraints_widgets['fingerprint_matching'].value}\n")
        config_file.write(f"14. Substructure == {constraints_widgets['substructure'].value}\n")
        config_file.write(f"15. Substructure exclusion == {constraints_widgets['substructure_exclusion'].value}\n")
        config_file.write(f"\nCombination type for molecules :: {combination_type.value}\n")
        config_file.write(f"Number of generations :: {generation_level.value}\n")
        config_file.write(f"Molecule format in output file :: {output_type.value}\n")
        config_file.write(f"Maximum files per folder :: {max_files.value}\n")
        config_file.write(f"Library name :: {library_name.value}\n")
    
    run_chemgenpy(building_blocks, combination_type, generation_level, output_type, max_files, library_name, directory)

def write_min_max_constraints(config_file, widgets_list):
    for key, widget in widgets_list.items():
        if key in ['bonds', 'atoms', 'molecular_weight', 'rings', 'aromatic_rings', 'nonaromatic_rings', 'single_bonds', 'double_bonds', 'triple_bonds']:
            min_value, max_value = widget.children
            if min_value.value == "None" and max_value.value == "None":
                config_file.write(f"2. Min and max no. of {key} == None\n")
            else:
                config_file.write(f"2. Min and max no. of {key} == ({min_value.value}, {max_value.value})\n")

def run_chemgenpy(building_blocks, combination_type, generation_level, output_type, max_files, library_name, directory):
    environment = widgets.Text(value='None', placeholder='Enter Environment name', description='Environment name', style=style)
    run_button = widgets.Button(description='Run Chemgenpy', layout=Layout(width='auto', border='solid 1px black'), style=style)

    run_button.on_click(lambda x: execute_chemgenpy(environment, directory))
    display(widgets.VBox([widgets.HTML("""Specify the virtual environment in which you want to run the library generator"""), environment, run_button]))

def execute_chemgenpy(environment, directory):
    if environment.value == 'None':
        env_command = ""
    else:
        env_command = f"source activate {environment.value} && "

    output_dir = directory.value if directory.value else "./output/"
    os.system(f"{env_command}chemgenpyshell -i config.dat -b building_blocks.dat -o {output_dir}")
    display_statistics_button()

def display_statistics_button():
    statistics_heading = widgets.HTML(value="""<font color=green><font size=5><b><u>Statistics</font>""")
    display(statistics_heading)
    statistics_intro = widgets.HTML("""Generate Statistics of the generated library""", layout=widgets.Layout(height='45px', width='90%', size='20'))
    stats_button = widgets.Button(description='Run Statistics', layout=Layout(width='20%', border='solid 1px black'), style=style)
    stats_button.on_click(run_statistics)
    display(widgets.VBox([statistics_intro, stats_button]))

def run_statistics(e):
    from chemgenpy.feasibility import generate_statistics
    generate_statistics()
    display_feasibility_button()

def display_feasibility_button():
    feasibility_heading = widgets.HTML(value="""<font color=green><font size=5><b><u>Synthetic Feasibility</font>""")
    display(feasibility_heading)
    feasibility_intro = widgets.HTML("""Test the synthetic feasibility of the generated library molecules""", layout=widgets.Layout(height='45px', width='90%', size='20'))
    feasibility_button = widgets.Button(description='Run Feasibility Analysis', layout=Layout(width='20%', border='solid 1px black'), style=style)
    feasibility_button.on_click(run_feasibility_analysis)
    display(widgets.VBox([feasibility_intro, feasibility_button]))

def run_feasibility_analysis(e):
    from chemgenpy.feasibility import feasibility
    feasibility()

config_builder()
