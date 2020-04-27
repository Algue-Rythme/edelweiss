# Edelweiss

Librairies nécessaires:
- pygraphviz dernière version (sur github)
- Pytorch
- PyGSP
- Numpy
- Matplotlib
- Python 3
- Python-louvain

Pour générer le graphe de confusion entre les classes, lancer:

python main_smooth.py --single

Ligne 202 dans main.py: donner le chemin du fichier novel.plk pour loader le backbone

Remarques: les options qui peuvent être modifiées concernant la création du graphe sont dans parse_grid.py
Ligne 36: grid.add_range('num_neighbors', [20])

Cela signifie que 20 voisins seront conservés.

Le code est compliqué et contient les cicatrices de nombreux brouillons.

Le code de confusion entre classe est la fonction 'monitore_communities' (ligne 237 du fichier monitoring.py)
Si tu veux étudier le code en détail tu peux commencer par là.
