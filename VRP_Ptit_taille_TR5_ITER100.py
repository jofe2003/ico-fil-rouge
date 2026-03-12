

import random
import math

# PARAMETRES 

SEED         = 42     # seed aleatoire fixe -> memes donnees a chaque execution
N_CLIENTS    = 20     
N_VEHICULES  = 3      
CAPACITE_Q   = 250    # capacite maximale de chaque vehicule
TAILLE_TABOU = 5      
NB_MAX_ITER  = 100    



# On genere aleatoirement :
# Le depot est toujours en position (50, 50)

def generer_donnees(n_clients, seed):
    random.seed(seed)

    
    coords = {0: (50, 50)}
    demandes = {0: 0}  # le depot n'a pas de demande

   
    for i in range(1, n_clients + 1):
        coords[i]  = (random.randint(0, 100), random.randint(0, 100))
        demandes[i] = random.randint(10, 40)

    return coords, demandes



# Distance euclidienne : d(i,j) = sqrt((xi-xj)^2 + (yi-yj)^2)


def distance(i, j, coords):
    xi, yi = coords[i]
    xj, yj = coords[j]
    return math.sqrt((xi - xj)**2 + (yi - yj)**2)


# Ici K(x) = 3 (fixe)

def calculer_cout(routes, coords):
    cout_total = 0
    for route in routes:
        for k in range(len(route) - 1):
            cout_total += distance(route[k], route[k+1], coords)
    return cout_total



def capacite_ok(route, demandes):
    # On ne compte pas le depot (indice 0)
    charge = sum(demandes[c] for c in route if c != 0)
    return charge <= CAPACITE_Q




# On repartit les clients aleatoirement dans les 3 routes
# en respectant la contrainte de capacite.


def solution_initiale(n_clients, n_vehicules, demandes, seed):
    random.seed(seed + 1)  # seed different de la generation des donnees

    clients = list(range(1, n_clients + 1))
    random.shuffle(clients)

    
    routes = [[0] for _ in range(n_vehicules)]
    charges = [0] * n_vehicules  
    for client in clients:
        place = False
        for v in range(n_vehicules):
            if charges[v] + demandes[client] <= CAPACITE_Q:
                routes[v].append(client)
                charges[v] += demandes[client]
                place = True
                break
        if not place:
            print(f"ATTENTION : client {client} n'a pas pu etre place !")

    # Ajouter le depot a la fin de chaque route
    for v in range(n_vehicules):
        routes[v].append(0)

    return routes


# =============================================================
# BLOC 6 : GENERATION DU VOISINAGE PAR RELOCATION
# =============================================================
# On genere tous les mouvements de relocation possibles.
# Un mouvement = (client, v_src, pos_src, v_dst, pos_dst)
#   client   : le client qu'on deplace
#   v_src    : indice de la route source
#   pos_src  : position du client dans la route source
#   v_dst    : indice de la route destination
#   pos_dst  : position ou on insere le client dans la route destination
#
# On ne deplace pas si la route source n'aurait plus qu'1 client
# (on ne veut pas de routes vides sauf si necessaire)

def generer_voisins(routes, demandes, coords):
    voisins = []

    for v_src in range(len(routes)):
        # Positions des clients dans la route source (pas le depot)
        clients_src = [(pos, c) for pos, c in enumerate(routes[v_src]) if c != 0]

        for pos_src, client in clients_src:

            for v_dst in range(len(routes)):
                if v_dst == v_src:
                    continue  # pas de relocation dans la meme route

                # Verifier que la route destination peut accueillir le client
                charge_dst = sum(demandes[c] for c in routes[v_dst] if c != 0)
                if charge_dst + demandes[client] > CAPACITE_Q:
                    continue  # capacite depassee -> mouvement invalide

                # Essayer toutes les positions d'insertion dans v_dst
                # (entre chaque paire de noeuds consecutifs, depot inclus)
                for pos_dst in range(1, len(routes[v_dst])):
                    voisins.append((client, v_src, pos_src, v_dst, pos_dst))

    return voisins


# =============================================================
# BLOC 7 : APPLIQUER UN MOUVEMENT DE RELOCATION
# =============================================================
# Retourne une COPIE des routes apres le mouvement
# (on ne modifie pas la solution courante directement)

def appliquer_mouvement(routes, mouvement):
    client, v_src, pos_src, v_dst, pos_dst = mouvement

    # Copie profonde des routes
    nouvelles_routes = [r[:] for r in routes]

    # Supprimer le client de la route source
    nouvelles_routes[v_src].pop(pos_src)

    # Inserer le client dans la route destination
    nouvelles_routes[v_dst].insert(pos_dst, client)

    return nouvelles_routes


# =============================================================
# BLOC 8 : MOUVEMENT INVERSE (pour la liste taboue)
# =============================================================
# Le mouvement inverse de (client, v_src, pos_src, v_dst, pos_dst)
# est (client, v_dst, ?, v_src, ?)
# On memorise juste (client, v_dst, v_src) :
#   "ce client ne doit pas revenir de v_dst vers v_src"

def mouvement_inverse(mouvement):
    client, v_src, pos_src, v_dst, pos_dst = mouvement
    # On memorise : (client, route_actuelle, route_interdite)
    return (client, v_dst, v_src)


# =============================================================
# BLOC 9 : VERIFICATION TABOU
# =============================================================
# Un mouvement est tabou si son inverse est dans la liste T
# i.e. si (client, v_dst, v_src) est dans T

def est_tabou(mouvement, liste_tabou):
    client, v_src, pos_src, v_dst, pos_dst = mouvement
    # Ce mouvement amenerait client de v_src vers v_dst
    # C'est tabou si (client, v_src, v_dst) est dans la liste
    return (client, v_src, v_dst) in liste_tabou


# =============================================================
# BLOC 10 : ALGORITHME TABOU PRINCIPAL
# =============================================================

def tabou_vrp(routes_init, coords, demandes):

    # ── Initialisation ──────────────────────────────────────
    routes_courantes = [r[:] for r in routes_init]
    cout_courant     = calculer_cout(routes_courantes, coords)

    routes_star = [r[:] for r in routes_courantes]  # meilleure solution
    cout_star   = cout_courant

    liste_tabou = []   # liste des mouvements tabous (inverses)
    nbiter      = 0
    meil_iter   = 0    # iteration ou on a trouve s*

    print(f"Depart : cout = {cout_courant:.2f}")
    print(f"Parametres : taille_tabou={TAILLE_TABOU}, nb_max_iter={NB_MAX_ITER}")
    print("-" * 60)

    # ── Boucle principale ────────────────────────────────────
    while nbiter < NB_MAX_ITER:
        nbiter += 1

        # ETAPE 1 : Generer tous les voisins (mouvements de relocation)
        voisins = generer_voisins(routes_courantes, demandes, coords)

        if not voisins:
            print("Plus de voisins disponibles -> arret")
            break

        # ETAPE 2 : Choisir le meilleur mouvement non-tabou (ou aspiration)
        meilleur_mouvement  = None
        meilleur_cout       = float('inf')
        meilleurs_routes    = None

        for mouvement in voisins:
            # Appliquer le mouvement et calculer le cout
            nouvelles_routes = appliquer_mouvement(routes_courantes, mouvement)
            nouveau_cout     = calculer_cout(nouvelles_routes, coords)

            tabou = est_tabou(mouvement, liste_tabou)

            # Critere d'aspiration : on leve le tabou si meilleur que s*
            aspiration = nouveau_cout < cout_star

            if (not tabou) or aspiration:
                if nouveau_cout < meilleur_cout:
                    meilleur_cout      = nouveau_cout
                    meilleur_mouvement = mouvement
                    meilleurs_routes   = nouvelles_routes

        
        if meilleur_mouvement is None:
            print(f"Iter {nbiter:>3} : aucun mouvement valide")
            continue

        
        routes_courantes = meilleurs_routes
        cout_courant     = meilleur_cout

       
        inv = mouvement_inverse(meilleur_mouvement)
        liste_tabou.append(inv)

        
        if len(liste_tabou) > TAILLE_TABOU:
            liste_tabou.pop(0)

        # ETAPE 6 : Mettre a jour la meilleure solution s*
        amelioration = ""
        if cout_courant < cout_star:
            cout_star     = cout_courant
            routes_star   = [r[:] for r in routes_courantes]
            meil_iter     = nbiter
            amelioration  = "  <-- nouveau meilleur !"

        # Affichage de l'iteration
        client_mv = meilleur_mouvement[0]
        v_src_mv  = meilleur_mouvement[1]
        v_dst_mv  = meilleur_mouvement[3]
        print(f"Iter {nbiter:>3} : cout = {cout_courant:>8.2f} "
              f"| C{client_mv} : R{v_src_mv+1}->R{v_dst_mv+1} "
              f"| tabou={liste_tabou}{amelioration}")

    print("-" * 60)
    print(f"Meilleure solution trouvee a l'iteration {meil_iter}")
    return routes_star, cout_star



def afficher_solution(routes, cout, coords, demandes, titre="Solution"):
    print(f"\n{'='*60}")
    print(f"{titre}")
    print(f"{'='*60}")
    print(f"Cout total : {cout:.2f}")
    print(f"Nombre de vehicules : {len(routes)}")
    print()
    for v, route in enumerate(routes):
        charge = sum(demandes[c] for c in route if c != 0)
        noms   = [f"C{c}" if c != 0 else "Depot" for c in route]
        print(f"  Vehicule {v+1} : {' -> '.join(noms)}")
        print(f"             Charge : {charge}/{CAPACITE_Q}")
    print()



print("=" * 60)
print("VRP - METHODE TABOU")
print("=" * 60)


coords, demandes = generer_donnees(N_CLIENTS, SEED)

print(f"\nDonnees generees (seed={SEED}) :")
print(f"  Depot : coords={coords[0]}")
print(f"  Clients :")
for i in range(1, N_CLIENTS + 1):
    print(f"    C{i:>2} : coords={coords[i]}, demande={demandes[i]}")

# ── Solution initiale 
routes_init = solution_initiale(N_CLIENTS, N_VEHICULES, demandes, SEED)
cout_init   = calculer_cout(routes_init, coords)
afficher_solution(routes_init, cout_init, coords, demandes,
                  "SOLUTION INITIALE (aleatoire)")


print("\nLANCEMENT DE L'ALGORITHME TABOU")
print("-" * 60)
routes_finale, cout_final = tabou_vrp(routes_init, coords, demandes)


afficher_solution(routes_finale, cout_final, coords, demandes,
                  "SOLUTION FINALE (tabou)")

print("=" * 60)
print("BILAN")
print("=" * 60)
print(f"  Cout initial  : {cout_init:.2f}")
print(f"  Cout final    : {cout_final:.2f}")
amelioration = ((cout_init - cout_final) / cout_init) * 100
print(f"  Amelioration  : {amelioration:.1f}%")
print(f"  Taille tabou  : {TAILLE_TABOU}")
print(f"  Nb iterations : {NB_MAX_ITER}")