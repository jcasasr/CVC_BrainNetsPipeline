import numpy as np
import copy
import pandas as pd
import os

"""
DUPLICACIO DE LES DADES MITJANÇANT PERMUTACIÓ DE LES PARTS DEL CERVELL
! Aquesta duplicació NOMÉS es fa servir per a entrenar al model donades les poques dades que tenim. 
! En CAP moment s'empraran per a comprovar si el model funciona bé, per a això SEMPRE s'usaran dades de pacients REALS.
"""

data = np.load('./data/data.npy')
patient_id = pd.read_csv('./data/ID_corr.csv')
folder = './node_volumes'

permutacions = [
    [0, 45], [1, 46], [2, 47], [3, 48], [4, 49], [5, 50], [6, 51], [7, 52],
    [8, 53], [9, 54], [10, 55], [11, 56], [12, 57], [13, 58], [14, 59], 
    [15, 60], [16, 61], [17, 62], [18, 63], [19, 64], [20, 65], [21, 66], 
    [22, 67], [23, 68], [24, 69], [25, 70], [26, 71], [27, 72], [28, 73], 
    [29, 74], [30, 75], [31, 38], [32, 39], [33, 40], [34, 41], [35, 42], 
    [36, 43], [37, 44]
]

brain_regions = [
    "ctx-lh-caudalanteriorcingulate", "ctx-lh-caudalmiddlefrontal", "ctx-lh-cuneus", 
    "ctx-lh-entorhinal", "ctx-lh-fusiform", "ctx-lh-inferiorparietal", 
    "ctx-lh-inferiortemporal", "ctx-lh-isthmuscingulate", "ctx-lh-lateraloccipital", 
    "ctx-lh-lateralorbitofrontal", "ctx-lh-lingual", "ctx-lh-medialorbitofrontal", 
    "ctx-lh-middletemporal", "ctx-lh-parahippocampal", "ctx-lh-paracentral", 
    "ctx-lh-parsopercularis", "ctx-lh-parsorbitalis", "ctx-lh-parstriangularis", 
    "ctx-lh-pericalcarine", "ctx-lh-postcentral", "ctx-lh-posteriorcingulate", 
    "ctx-lh-precentral", "ctx-lh-precuneus", "ctx-lh-rostralanteriorcingulate", 
    "ctx-lh-rostralmiddlefrontal", "ctx-lh-superiorfrontal", "ctx-lh-superiorparietal", 
    "ctx-lh-superiortemporal", "ctx-lh-supramarginal", "ctx-lh-transversetemporal", 
    "ctx-lh-insula", "Left-Thalamus-Proper", "Left-Caudate", "Left-Putamen", 
    "Left-Pallidum", "Left-Hippocampus", "Left-Amygdala", "Left-Accumbens-area", 
    "Right-Thalamus-Proper", "Right-Caudate", "Right-Putamen", "Right-Pallidum", 
    "Right-Hippocampus", "Right-Amygdala", "Right-Accumbens-area", 
    "ctx-rh-caudalanteriorcingulate", "ctx-rh-caudalmiddlefrontal", "ctx-rh-cuneus", 
    "ctx-rh-entorhinal", "ctx-rh-fusiform", "ctx-rh-inferiorparietal", 
    "ctx-rh-inferiortemporal", "ctx-rh-isthmuscingulate", "ctx-rh-lateraloccipital", 
    "ctx-rh-lateralorbitofrontal", "ctx-rh-lingual", "ctx-rh-medialorbitofrontal", 
    "ctx-rh-middletemporal", "ctx-rh-parahippocampal", "ctx-rh-paracentral", 
    "ctx-rh-parsopercularis", "ctx-rh-parsorbitalis", "ctx-rh-parstriangularis", 
    "ctx-rh-pericalcarine", "ctx-rh-postcentral", "ctx-rh-posteriorcingulate", 
    "ctx-rh-precentral", "ctx-rh-precuneus", "ctx-rh-rostralanteriorcingulate", 
    "ctx-rh-rostralmiddlefrontal", "ctx-rh-superiorfrontal", "ctx-rh-superiorparietal", 
    "ctx-rh-superiortemporal", "ctx-rh-supramarginal", "ctx-rh-transversetemporal", 
    "ctx-rh-insula"
]

def csvlist_to_matrix(folder = './node_volumes'):
    """"
    ara mateix a /node_volumes tenim els 270 patients amb els nodeweights normalitzats,
    passarem a tenirho tot a llistes per a poder tractar-ho millor, és a dir:
    el patient 0 passarà de tenir un csv a ser l'index [0], el qual contindrà una array de 76 pos, amb els pesos de cada part del cervell
    """
    valuelist = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        file = pd.read_csv(file_path, header=None) # no hi ha header, si no es posa aixo ignora el primer valor
        file = file[0].to_numpy() # pillem la columna de valors
        valuelist.append(file)

    return np.array(valuelist)


def girar_duplicar(data, folder='./node_volumes'):
    """
    passem de tenir X dades a tenir-ne X*2, aplicant
    les transofmracions de la llista de permutacions
    """

    weight_matrix = csvlist_to_matrix(folder)

    weight_matrix_girada = copy.deepcopy(weight_matrix)
    data_girada = copy.deepcopy(data) # pq els canvis no es reflecteixin en laltre
    
    for p in permutacions:
        data_girada[:, p, :, :] = data_girada[:, list(reversed(p)), :, :]
        data_girada[:, :, p, :] = data_girada[:, :, list(reversed(p)), :]
        weight_matrix_girada[:, [p[0], p[1]]] = weight_matrix_girada[:, [p[1], p[0]]]
    
    weight_matrix_combinat = np.concatenate((weight_matrix, weight_matrix_girada), axis=0)
    data_combinada = np.concatenate((data, data_girada), axis=0) # axis 0 apila per files, que es el que ens interessa per conservar les columnes
    
    #guardem els dos fitxers
    np.save('./data/data_combinada.npy', data_combinada)
    weight_df = pd.DataFrame(weight_matrix_combinat, columns=brain_regions)
    weight_df.to_csv('./data/nodeweights_combinats.csv')

    return data_combinada

d = girar_duplicar(data)
# print(d.shape) # hauria de tornar 540, 76, 76, 3
