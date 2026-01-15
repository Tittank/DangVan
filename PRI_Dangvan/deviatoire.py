import numpy as np

def CalculMatDev(matrice_tenseurs):
    """
    Calcule le tenseur deviatoire pour chaque tenseur dans la matrice.
    
    Args:
        matrice_tenseurs (list of lists): Matrice où chaque ligne est un tenseur
                                         exprimé comme une liste de 9 éléments (3x3).
    
    Returns:
        list of lists: Matrice des tenseurs deviatoires, chaque ligne étant
                       une liste de 9 éléments.
    """
    result = []
    for tenseur in matrice_tenseurs:
        # Reshape la liste en matrice 3x3
        sigma = np.array(tenseur).reshape(3, 3)
        # Calcul de la trace
        trace = np.trace(sigma)
        # Tenseur hydrostatique
        hydrostatic = (trace / 3) * np.eye(3)
        # Tenseur deviatoire
        deviator = sigma - hydrostatic
        # Aplatir et ajouter à la liste résultat
        result.append(deviator.flatten().tolist())
    return result

def CalculNormeTresca(input_tenseur):
    """
    Calcule la norme de Tresca pour un tenseur ou un vecteur de contraintes principales.

    Args:
        input_tenseur (list): Soit une liste de 9 éléments (tenseur 3x3 aplati),
                              soit une liste de 3 éléments (contraintes principales).

    Returns:
        float: La norme de Tresca (différence entre la plus grande et la plus petite
               contrainte principale).
    """
    if len(input_tenseur) == 9:
        # Traiter comme un tenseur 3x3
        sigma = np.array(input_tenseur).reshape(3, 3)
        eigenvals = np.linalg.eigvals(sigma)
    elif len(input_tenseur) == 3:
        # Traiter comme les contraintes principales
        eigenvals = np.array(input_tenseur)
    else:
        raise ValueError("L'entrée doit être une liste de 3 ou 9 éléments.")

    # Norme de Tresca : max - min des valeurs propres
    return np.max(eigenvals) - np.min(eigenvals)