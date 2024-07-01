import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import csv
from collections import Counter
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import base64
import io
from jupyter_dash import JupyterDash

# Initialize the JupyterDash app
app = JupyterDash(__name__)

app.layout = html.Div([
    html.H1("Molecular Analysis Tool"),

    html.H2("Feasibility Analysis"),
    html.Label("Cut-off Value"),
    dcc.Input(id='cutoff-value', type='text', placeholder="Enter the cut off value of Tanimoto index"),
    html.Button('Confirm cut-off', id='cutoff-button'),
    html.Div(id='feasibility-output'),

    html.H2("Generate Statistics"),
    html.Button('Generate Statistics', id='stats-button'),
    html.Div(id='stats-output')
])

@app.callback(
    Output('feasibility-output', 'children'),
    [Input('cutoff-button', 'n_clicks')],
    [State('cutoff-value', 'value')]
)
def feasibility(n_clicks, cutoff_value):
    if n_clicks is None or cutoff_value is None:
        return ""

    with open('database.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        commercial = [row[0] for idx, row in enumerate(csv_reader) if idx != 0]

    with open('final_smiles.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        library = [row[0] for row in csv_reader]

    final = []
    tanimoto1 = []
    fll = []

    cutoff_value = float(cutoff_value)
    for i in library:
        mol1 = Chem.MolFromSmiles(i)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        for mol in commercial:
            mol2 = Chem.MolFromSmiles(mol)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)

            if tanimoto >= cutoff_value:
                tanimoto1.append(tanimoto)
                final.append(mol)
                fll.append(i)

    data = {'Molecule': final, 'Tanimoto': tanimoto1, 'Final library': fll}
    df = pd.DataFrame(data)
    df.to_excel("output.xlsx")

    grouped = df.groupby(['Final library']).count()[['Molecule']]
    sorted_df = grouped.sort_values(by='Molecule', ascending=False)
    sorted_df.to_excel("dataframe.xlsx")

    return "Feasibility analysis complete. Results saved to output.xlsx and dataframe.xlsx"

@app.callback(
    Output('stats-output', 'children'),
    [Input('stats-button', 'n_clicks')]
)
def generate_statistics(n_clicks):
    if n_clicks is None:
        return ""

    with open('final_library.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        library = []
        smiles = []
        for row in csv_reader:
            library.append(row[0])
            smiles.append(row[1])

    building_blocks = [bb.split('-') for bb in library]
    number_of_building_blocks = [dict(Counter(block)) for block in building_blocks]

    df = pd.DataFrame(number_of_building_blocks, index=smiles)
    df.fillna(0, inplace=True)
    df['Total building blocks'] = df.sum(axis=1, numeric_only=True)
    df.to_excel("statistics.xlsx")

    return "Statistics generated and saved to statistics.xlsx"

if __name__ == '__main__':
    app.run_server(mode='inline', debug=True)

