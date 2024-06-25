import csv
import pandas as pd
from collections import Counter
import pybel
from ipywidgets import interact, Layout, Label
import ipywidgets as widgets
from IPython.display import display, clear_output

style = {'description_width': 'initial', 'font_weight': 'bold'}

def read_csv_file(file_path, delimiter='\t'):
    with open(file_path, mode='r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        data = [row for row in reader]
    return data

def read_commercial_database(file_path):
    data = read_csv_file(file_path, delimiter='\t')
    return [row[0] for row in data[1:]]

def read_library(file_path):
    data = read_csv_file(file_path, delimiter=',')
    return [row[0] for row in data]

def calculate_tanimoto(library, commercial, cutoff_value):
    final = []
    tanimoto1 = []
    fll = []
    for i in library:
        fl = pybel.readstring('smi', i)
        for mol in commercial:
            smiles = pybel.readstring('smi', mol)
            tanimoto = smiles.calcfp() | fl.calcfp()
            if tanimoto >= cutoff_value:
                tanimoto1.append(tanimoto)
                final.append(mol)
                fll.append(i)
    return final, tanimoto1, fll

def feasibility():
    commercial = read_commercial_database('database.txt')
    library = read_library('final_smiles.csv')
    
    cutoff = widgets.Text(description="Cut-off Value", placeholder="Enter the cut off value of Tanimoto index", style=style)
    cutoff_button = widgets.Button(description='Confirm cut-off', layout=Layout(width='35%', border='solid 1px black'))
    display(cutoff)
    display(cutoff_button)

    def cutoff_clicked(m):
        clear_output(wait=True)
        display(cutoff)
        display(cutoff_button)
        try:
            cutoff_value = float(cutoff.value)
        except ValueError:
            print("Please enter a valid numeric value for the cutoff.")
            return

        final, tanimoto1, fll = calculate_tanimoto(library, commercial, cutoff_value)

        data = {'Molecule': final, 'Tanimoto': tanimoto1, 'Final library': fll}
        df = pd.DataFrame(data)
        pd.set_option('display.max_colwidth', None)

        grouped = df.groupby(['Final library']).count()[['Molecule']]
        sorted_df = grouped.sort_values(by='Molecule', ascending=False)
        sorted_df.to_excel("dataframe.xlsx")
        df.to_excel("output.xlsx")
        display(sorted_df)

    cutoff_button.on_click(cutoff_clicked)

def generate_statistics():
    data = read_csv_file('final_library.csv', delimiter=',')
    library = [row[0] for row in data[1:]]
    smiles = [row[1] for row in data[1:]]

    building_blocks = [bb.split('-') for bb in library]
    number_of_building_blocks = [dict(Counter(block)) for block in building_blocks]

    df = pd.DataFrame(number_of_building_blocks, index=smiles)
    df.fillna(0, inplace=True)
    df['Total building blocks'] = df.sum(axis=1, numeric_only=True)
    pd.set_option('display.max_colwidth', None)
    df.to_excel("statistics.xlsx")
    display(df)

# Call the functions to ensure they work as intended
#generate_statistics()
#feasibility()
