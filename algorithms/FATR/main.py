import sys
sys.path.append('./pyten/')

import numpy as np
import pandas as pd
import pyten.tenclass
from FATR_Newton import FATR_Newton

def prepare_data_from_matrix(rating_matrix, user_genders):
    """
    Converte una matrice 2D completa (users x features) in formato tensor 3D per FATR_Newton
    
    :param rating_matrix: numpy array (n_users, n_features) con i rating - COMPLETAMENTE OSSERVATA
    :param user_genders: numpy array (n_users,) con 0=maschio, 1=femmina
    :return: tutti gli input necessari per FATR_Newton
    """
    
    n_users, n_features = rating_matrix.shape
    n_genders = 2  # maschio e femmina
    
    print "="*50
    print "PREPARAZIONE DATI"
    print "="*50
    print "Rating matrix shape:", rating_matrix.shape
    print "User genders shape:", user_genders.shape
    print "Number of users:", n_users
    print "Number of features:", n_features
    print "Matrix min/max:", np.min(rating_matrix), np.max(rating_matrix)
    
    # Verifica user_genders
    unique_genders = np.unique(user_genders)
    print "\nGender distribution:"
    print "Unique genders:", unique_genders
    print "Males (0):", np.sum(user_genders == 0)
    print "Females (1):", np.sum(user_genders == 1)
    
    # Crea tensor 3D: (users x features x 1)
    # La terza dimensione è fittizia ma necessaria per il formato 3D
    tensor_shape = (n_users, n_features, 1)
    rating_tensor = rating_matrix.reshape(tensor_shape)
    
    print "\nTensor shape:", tensor_shape
    
    # Crea omega mask - TUTTA A 1 perché non ci sono dati mancanti
    omega = np.ones(tensor_shape)
    
    print "All entries observed (omega=1):", np.sum(omega)
    
    # Crea features per gli USERS (gender è associato agli utenti)
    feature_d = 0  # dimensione 0 = users
    feature_n = n_genders
    
    # One-hot encoding del genere
    features = np.zeros((n_users, n_genders))
    for i in range(n_users):
        gender = int(user_genders[i])
        features[i, gender] = 1
    
    print "\nFeatures (gender encoding):"
    print "  Shape:", features.shape
    print "  Males (column 0):", np.sum(features[:, 0])
    print "  Females (column 1):", np.sum(features[:, 1])
    
    # Crea omega_groups (maschere per gruppo di genere)
    omega_groups = []
    
    # Gruppo 0: maschi
    mask_male = np.zeros(tensor_shape)
    male_user_indices = np.where(user_genders == 0)[0]
    for user_idx in male_user_indices:
        mask_male[user_idx, :, :] = omega[user_idx, :, :]
    omega_groups.append(mask_male)
    print "\nGroup 0 (Males): {0} entries".format(int(np.sum(mask_male)))
    
    # Gruppo 1: femmine
    mask_female = np.zeros(tensor_shape)
    female_user_indices = np.where(user_genders == 1)[0]
    for user_idx in female_user_indices:
        mask_female[user_idx, :, :] = omega[user_idx, :, :]
    omega_groups.append(mask_female)
    print "Group 1 (Females): {0} entries".format(int(np.sum(mask_female)))
    
    # Verifica che entrambi i gruppi abbiano dati
    if np.sum(omega_groups[0]) == 0:
        raise ValueError("Il gruppo maschi non ha dati!")
    if np.sum(omega_groups[1]) == 0:
        raise ValueError("Il gruppo femmine non ha dati!")
    
    # Converti a pyten Tensor
    y = pyten.tenclass.Tensor(rating_tensor)
    
    print "\nDati pronti per FATR_Newton!"
    
    return y, features, feature_d, feature_n, omega, omega_groups


def main_from_matrix(rating_matrix, user_genders, r=20, reg_para=0.1, 
                     Freg_para=0.1, tol=1e-4, maxiter=500, printitn=10):
    """
    Main function che parte da matrice numpy completa e vettore generi
    
    :param rating_matrix: numpy array (n_users, n_features) - completamente osservata
    :param user_genders: numpy array (n_users,) con 0=maschio, 1=femmina
    :param r: rank del tensor
    :param reg_para: parametro di regolarizzazione (lambda=0.00001)
    :param Freg_para: parametro di regolarizzazione Frobenius (gamma=0.01)
    :param tol: tolleranza per convergenza
    :param maxiter: numero massimo di iterazioni
    :param printitn: stampa ogni n iterazioni
    """
    
    # 1. Prepara i dati nel formato corretto
    y, features, feature_d, feature_n, omega, omega_groups = \
        prepare_data_from_matrix(rating_matrix, user_genders)
    
    # 2. Parametri
    print "\n" + "="*50
    print "PARAMETRI ALGORITMO"
    print "="*50
    print "Rank (r):", r
    print "Regularization (reg_para):", reg_para
    print "Frobenius regularization (Freg_para):", Freg_para
    print "Max iterations:", maxiter
    print "Tolerance:", tol
    print "Sensitive attribute: User Gender (dimension 0)"
    print ""
    
    # 3. Esegui FATR_Newton
    print "="*50
    print "AVVIO ALGORITMO FATR_Newton"
    print "="*50
    print ""
    
    U, Xf, X = FATR_Newton(
        y=y,
        features=features,
        feature_d=feature_d,
        feature_n=feature_n,
        r=r,
        omega=omega,
        omega_groups=omega_groups,
        reg_para=reg_para,
        Freg_para=Freg_para,
        tol=tol,
        maxiter=maxiter,
        init='random',
        printitn=printitn
    )
    
    print "\n" + "="*50
    print "ALGORITMO COMPLETATO"
    print "="*50
    
    # 4. Estrai la matrice 2D dai risultati
    # Il tensor fair è 3D, lo riconvertiamo a 2D
    fair_matrix = Xf.data.reshape(rating_matrix.shape)
    recovered_matrix = X.data.reshape(rating_matrix.shape)
    
    print "\nRisultati:"
    print "  Fair matrix shape:", fair_matrix.shape
    print "  Recovered matrix shape:", recovered_matrix.shape
    
    # 5. Verifica risultati
    print "\n" + "="*50
    print "VERIFICA RISULTATI"
    print "="*50
    print "Fair matrix:"
    print "  min/max:", np.min(fair_matrix), "/", np.max(fair_matrix)
    print "  mean:", np.mean(fair_matrix)
    print "  has NaN?", np.isnan(fair_matrix).any()
    print "  has Inf?", np.isinf(fair_matrix).any()
    
    print "\nRecovered matrix:"
    print "  min/max:", np.min(recovered_matrix), "/", np.max(recovered_matrix)
    print "  mean:", np.mean(recovered_matrix)
    print "  has NaN?", np.isnan(recovered_matrix).any()
    print "  has Inf?", np.isinf(recovered_matrix).any()
    
    # 6. Calcola la differenza con la matrice originale
    diff_fair = np.linalg.norm(fair_matrix - rating_matrix)
    diff_recovered = np.linalg.norm(recovered_matrix - rating_matrix)
    
    print "\nDifferenza dalla matrice originale:"
    print "  Fair matrix (Frobenius norm):", diff_fair
    print "  Recovered matrix (Frobenius norm):", diff_recovered
    
    # 7. Analisi fairness
    male_indices = np.where(user_genders == 0)[0]
    female_indices = np.where(user_genders == 1)[0]
    
    print "\n" + "="*50
    print "ANALISI FAIRNESS"
    print "="*50
    print "Matrice originale:"
    print "  Mean rating maschi:", np.mean(rating_matrix[male_indices, :])
    print "  Mean rating femmine:", np.mean(rating_matrix[female_indices, :])
    print "  Differenza:", abs(np.mean(rating_matrix[male_indices, :]) - 
                              np.mean(rating_matrix[female_indices, :]))
    
    print "\nMatrice fair:"
    print "  Mean rating maschi:", np.mean(fair_matrix[male_indices, :])
    print "  Mean rating femmine:", np.mean(fair_matrix[female_indices, :])
    print "  Differenza:", abs(np.mean(fair_matrix[male_indices, :]) - 
                              np.mean(fair_matrix[female_indices, :]))
    
    # 8. Salva risultati
    print "\n" + "="*50
    print "SALVATAGGIO RISULTATI"
    print "="*50
    np.save('fair_matrix.npy', fair_matrix)
    np.save('recovered_matrix.npy', recovered_matrix)
    np.save('U_users.npy', U[0])
    np.save('U_features.npy', U[1])
    
    print "Salvati:"
    print "  - fair_matrix.npy: matrice fair ({0})".format(fair_matrix.shape)
    print "  - recovered_matrix.npy: matrice recuperata ({0})".format(recovered_matrix.shape)
    print "  - U_users.npy: fattori utenti ({0})".format(U[0].shape)
    print "  - U_features.npy: fattori features ({0})".format(U[1].shape)
    
    return fair_matrix, recovered_matrix, U


