import os
import time
import tracemalloc
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx
from scipy.stats import kendalltau

# ==============================================================================
# INCISO A: GENERACIÓN DE GRAFOS Y CONSTRUCCIÓN DE MATRICES S Y M_alpha
# ==============================================================================
def generar_grafo_sintetico(tipo, n, con_dangling=True, seed=42):
    if tipo == 'ER':
        G = nx.erdos_renyi_graph(n, p=4.0/n, directed=True, seed=seed)
    elif tipo == 'BA':
        G_und = nx.barabasi_albert_graph(n, m=2, seed=seed)
        G = nx.DiGraph()
        G.add_nodes_from(G_und.nodes())
        for u, v in G_und.edges():
            if u > v:
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)
    else:
        raise ValueError("Tipo debe ser 'ER' o 'BA'")

    if not con_dangling:
        for nodo in list(G.nodes()):
            if G.out_degree(nodo) == 0:
                G.add_edge(nodo, nodo)
    else:
        if all(G.out_degree(nodo) > 0 for nodo in G.nodes()):
            nodo_obj = list(G.nodes())[0]
            G.remove_edges_from(list(G.out_edges(nodo_obj)))

    return G

def cargar_grafo_snap(ruta_txt=r"C:\repositorios_ucsp\MAC\web-Stanford.txt", max_nodos=2000):
    G_completo = nx.read_edgelist(
        ruta_txt,
        comments="#",
        create_using=nx.DiGraph(),
        nodetype=int
    )
    if max_nodos is not None and G_completo.number_of_nodes() > max_nodos:
        nodo_inicio = next(iter(G_completo.nodes()))
        nodos_sub = list(nx.bfs_tree(G_completo, nodo_inicio).nodes())[:max_nodos]
        if len(nodos_sub) < max_nodos:
            set_sub = set(nodos_sub)
            faltantes = [u for u in G_completo.nodes() if u not in set_sub]
            nodos_sub.extend(faltantes[:max_nodos - len(nodos_sub)])
        G = G_completo.subgraph(nodos_sub).copy()
    else:
        G = G_completo

    return nx.convert_node_labels_to_integers(G)

def construir_matrices(G, alpha=0.85, construir_densa=True):
    n = G.number_of_nodes()
    A_std = nx.to_scipy_sparse_array(G, format='csc', dtype=float)
    A = A_std.T.tocsc()

    d = np.array(A.sum(axis=0)).flatten()
    D = (d == 0)

    d_inv = np.zeros(n, dtype=float)
    d_inv[~D] = 1.0 / d[~D]
    S_base = A @ sp.diags(d_inv, format='csc')

    def matvec_M(v):
        dang_sum = np.sum(v[D]) / n
        return v - alpha * (S_base @ v + dang_sum)

    def rmatvec_M(v):
        res = v - alpha * (S_base.T @ v)
        res[D] -= alpha * (np.sum(v) / n)
        return res

    M_linop = spla.LinearOperator((n, n), matvec=matvec_M, rmatvec=rmatvec_M, dtype=float)

    if np.any(D):
        col_D = np.zeros(n, dtype=float)
        col_D[D] = 1.0 / n
        S_dispersa = (S_base + sp.csc_matrix(np.outer(np.ones(n), col_D))).tocsc()
    else:
        S_dispersa = S_base.tocsc()

    M_dispersa = (sp.eye(n, format='csc', dtype=float) - alpha * S_dispersa).tocsc()

    S_densa = S_dispersa.toarray() if construir_densa else None
    M_densa = M_dispersa.toarray() if construir_densa else None

    return {
        "n": n,
        "D": D,
        "S_base": S_base,
        "S_dispersa": S_dispersa,
        "M_dispersa": M_dispersa,
        "M_linop": M_linop,
        "S_densa": S_densa,
        "M_densa": M_densa
    }

# ==============================================================================
# INCISO B: IMPLEMENTACIÓN DE LOS CUATRO MÉTODOS
# ==============================================================================
def metodo_potencia(S_base, D, alpha=0.85, tol=1e-12, max_iter=15000):
    n = S_base.shape[0]
    pi = np.ones(n, dtype=float) / n
    for _ in range(max_iter):
        dang_mass = np.sum(pi[D]) / n
        pi_next = alpha * (S_base @ pi + dang_mass) + (1.0 - alpha) / n
        if np.linalg.norm(pi_next - pi, 1) < tol:
            return pi_next / np.sum(pi_next)
        pi = pi_next
    return pi / np.sum(pi)

def metodo_pinv(M_densa, alpha=0.85):
    n = M_densa.shape[0]
    b = np.full(n, (1.0 - alpha) / n)
    x = np.linalg.pinv(M_densa) @ b
    return x / np.linalg.norm(x, 1)

def metodo_svd_truncada(M_densa, k, alpha=0.85, svd_precalculada=None):
    n = M_densa.shape[0]
    b = np.full(n, (1.0 - alpha) / n)
    if svd_precalculada is None:
        U, s, Vt = np.linalg.svd(M_densa, full_matrices=False)
    else:
        U, s, Vt = svd_precalculada
    U_k = U[:, :k]
    s_inv_k = 1.0 / s[:k]
    V_k = Vt[:k, :].T
    x_k = V_k @ (s_inv_k * (U_k.T @ b))
    return x_k / np.linalg.norm(x_k, 1)

def metodo_svds_disperso(M_linop, k, alpha=0.85):
    n = M_linop.shape[0]
    b = np.full(n, (1.0 - alpha) / n)
    k_valido = min(k, n - 2)
    u, s, vt = spla.svds(M_linop, k=k_valido, which='LM')
    idx = np.argsort(s)[::-1]
    u, s, vt = u[:, idx], s[idx], vt[idx, :]
    x_k = vt.T @ ((1.0 / s) * (u.T @ b))
    return x_k / np.linalg.norm(x_k, 1)

# ==============================================================================
# MÉTRICAS AUXILIARES (INCISOS C, D, E, F)
# ==============================================================================
def precision_at_m(pi_true, pi_approx, m):
    top_true = set(np.argsort(pi_true)[::-1][:m])
    top_approx = set(np.argsort(pi_approx)[::-1][:m])
    return len(top_true & top_approx) / m

def medir_tiempo_memoria(func, *args, repeticiones=5):
    tiempos = []
    mem_pico = 0
    res = None
    for _ in range(repeticiones):
        tracemalloc.start()
        t0 = time.perf_counter()
        res = func(*args)
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tiempos.append(t1 - t0)
        mem_pico = max(mem_pico, peak)
    return res, float(np.mean(tiempos)), mem_pico / (1024 ** 2)

# ------ (E) Analisis por nodo

def analizar_nodos(G, pi_ref, pi_k, k):
    import matplotlib.pyplot as plt

    orden_ref = np.argsort(pi_ref)[::-1]
    orden_k = np.argsort(pi_k)[::-1]

    pos_ref = np.argsort(orden_ref)
    pos_k = np.argsort(orden_k)

    cambio_pos = np.abs(pos_ref - pos_k)
    cambio_pi = np.abs(pi_ref - pi_k)

    entrada = dict(G.in_degree())
    salida = dict(G.out_degree())
    puentes = nx.betweenness_centrality(G)

    print("\n=== INCISO E: NODOS MAS AFECTADOS ===")
    print("Nodo | PageRank | Cambio posicion | Entrada | Salida | Betweenness")

    afectados = np.argsort(cambio_pos)[::-1][:15]

    for nodo in afectados:
        print(
            f"{nodo:4d} | {pi_ref[nodo]:.6f} | "
            f"{cambio_pos[nodo]:5d} | "
            f"{entrada[nodo]:4d} | {salida[nodo]:4d} | "
            f"{puentes[nodo]:.5f}"
        )

    # Visualización del grafo
    posiciones = nx.spring_layout(G, seed=42)

    tamanos = 100 + 5000 * pi_ref
    colores = cambio_pos

    plt.figure(figsize=(12, 9))

    nx.draw_networkx(
        G,
        pos=posiciones,
        node_size=tamanos,
        node_color=colores,
        cmap=plt.cm.coolwarm,
        with_labels=False,
        arrows=True,
        alpha=0.75
    )

    plt.colorbar(
        plt.cm.ScalarMappable(
            cmap=plt.cm.coolwarm,
            norm=plt.Normalize(vmin=0, vmax=max(1, cambio_pos.max()))
        ),
        label="Cambio de posición"
    )

    plt.title(f"Análisis de nodos - SVD truncada k={k}")
    plt.savefig("inciso_e_grafo.png", dpi=300)
    plt.show()

    return cambio_pos, cambio_pi

# ------ (G) Perturbar las aristas
def perturbar_grafo(G, porcentaje=0.01, seed=42):
    import random

    G_nuevo = G.copy()
    rng = random.Random(seed)

    aristas = list(G_nuevo.edges())

    cantidad = max(1, int(round(len(aristas) * porcentaje)))

    # Eliminar el 1% de las aristas
    eliminar = rng.sample(aristas, min(cantidad, len(aristas)))

    G_nuevo.remove_edges_from(eliminar)

    # Agregar nuevas aristas
    nodos = list(G_nuevo.nodes())
    existentes = set(G_nuevo.edges())

    agregadas = 0
    intentos = 0

    while agregadas < len(eliminar) and intentos < cantidad * 100:
        origen = rng.choice(nodos)
        destino = rng.choice(nodos)

        if (origen, destino) not in existentes:
            G_nuevo.add_edge(origen, destino)
            existentes.add((origen, destino))
            agregadas += 1

        intentos += 1

    return G_nuevo

# ==============================================================================
# BLOQUE PRINCIPAL DE EJECUCIÓN
# ==============================================================================
if __name__ == "__main__":
    print("=== 1. INCISO A: GENERACIÓN DE GRAFOS Y MATRICES ===")
    tamanos = [100, 500, 1000, 2000, 5000]
    grafos = {}

    for tipo in ['ER', 'BA']:
        for n in tamanos:
            for con_d in [True, False]:
                nombre = f"{tipo}_n{n}_{'conD' if con_d else 'sinD'}"
                G = generar_grafo_sintetico(tipo, n, con_dangling=con_d)
                datos = construir_matrices(G, alpha=0.85, construir_densa=(n <= 2000))
                grafos[nombre] = (G, datos)
                print(f"[OK] {nombre:16s} | Dangling: {np.sum(datos['D']):4d} | S_disp nnz: {datos['S_dispersa'].nnz}")

    if os.path.exists(r"C:\repositorios_ucsp\MAC\web-Stanford.txt"):
        G_snap = cargar_grafo_snap(r"C:\repositorios_ucsp\MAC\web-Stanford.txt", max_nodos=1500)
        datos_snap = construir_matrices(G_snap, alpha=0.85, construir_densa=True)
        grafos["SNAP_sub1500"] = (G_snap, datos_snap)
        print(f"[OK] {'SNAP_sub1500':16s} | Dangling: {np.sum(datos_snap['D']):4d} | S_disp nnz: {datos_snap['S_dispersa'].nnz}")
    else:
        print("[AVISO] Coloca 'web-Stanford.txt' en C:\\repositorios_ucsp\\MAC\\ para incluir SNAP.")

    print("\n=== 2. INCISO B: VALIDACIÓN DE MÉTODOS (TOLERANCIA 10^-8) ===")
    _, d_val = grafos["BA_n100_conD"]
    pi_pot = metodo_potencia(d_val["S_base"], d_val["D"], alpha=0.85)
    pi_pinv = metodo_pinv(d_val["M_densa"], alpha=0.85)
    pi_svd = metodo_svd_truncada(d_val["M_densa"], k=100, alpha=0.85)

    err_pinv = np.linalg.norm(pi_pot - pi_pinv, 1)
    err_svd = np.linalg.norm(pi_pot - pi_svd, 1)
    print(f"Error L1 (Potencia vs Pinv):      {err_pinv:.2e}")
    print(f"Error L1 (Potencia vs SVD k=n):   {err_svd:.2e}")
    assert err_pinv < 1e-8 and err_svd < 1e-8, "Fallo en tolerancia 10^-8"
    print("-> Validación superada con éxito (< 10^-8).")

    print("\n=== 3. INCISO C: ESPECTRO Y CONDICIONAMIENTO VS ALPHA ===")
    G_c, _ = grafos["BA_n500_conD"]
    for a in [0.5, 0.85, 0.95, 0.99, 0.999]:
        d_a = construir_matrices(G_c, alpha=a, construir_densa=True)
        _, s, _ = np.linalg.svd(d_a["M_densa"], full_matrices=False)
        k2 = s[0] / s[-1]
        r_1e2 = np.sum(s > 1e-2 * s[0])
        r_1e4 = np.sum(s > 1e-4 * s[0])
        print(f"alpha={a:5.3f} | sigma_min={s[-1]:.5f} | kappa_2={k2:8.2f} | rango_ef(1e-2)={r_1e2:3d} | rango_ef(1e-4)={r_1e4:3d}")

        print("\n=== INCISO D: ERROR Y CALIDAD DEL RANKING ===")

    G_d, _ = grafos["BA_n500_conD"]

    alphas = [0.5, 0.85, 0.95, 0.99, 0.999]
    valores_k = [10, 25, 50, 100, 150, 200, 300, 400, 500]

    resultados_d = []

    for alpha in alphas:

        datos = construir_matrices(G_d, alpha, True)

        pi_ref = metodo_potencia(
            datos["S_base"], datos["D"], alpha
        )

        pi_pinv = metodo_pinv(datos["M_densa"], alpha)

        svd_cache = np.linalg.svd(
            datos["M_densa"], full_matrices=False
        )

        k_minimo = None

        for k in valores_k:

            pi_k = metodo_svd_truncada(
                datos["M_densa"], k, alpha, svd_cache
            )

            error = np.linalg.norm(pi_ref - pi_k, 1)
            tau, _ = kendalltau(pi_ref, pi_k)

            p10 = precision_at_m(pi_ref, pi_k, 10)
            p50 = precision_at_m(pi_ref, pi_k, 50)
            p100 = precision_at_m(pi_ref, pi_k, 100)

            resultados_d.append(
                (alpha, k, error, tau, p10, p50, p100)
            )

            if tau > 0.95 and k_minimo is None:
                k_minimo = k

            print(
                f"alpha={alpha} | k={k} | "
                f"L1={error:.6f} | Tau={tau:.5f} | "
                f"P10={p10:.2f} | P50={p50:.2f} | "
                f"P100={p100:.2f}"
            )

        print(f"--> K minimo para alpha={alpha}: {k_minimo}")

        # Comparación con los métodos de referencia
        print(
            "Error potencia vs pinv:",
            np.linalg.norm(pi_ref - pi_pinv, 1)
        )

        # SVD dispersa
        pi_sparse = metodo_svds_disperso(
            datos["M_linop"], 50, alpha
        )

        tau_sparse, _ = kendalltau(pi_ref, pi_sparse)

        print(f"Kendall Tau SVD dispersa k=50: {tau_sparse:.5f}")

        print("\n=== INCISO E: ANALISIS POR NODO ===")

    G_e, datos_e = grafos["BA_n500_conD"]

    pi_original = metodo_potencia(
        datos_e["S_base"],
        datos_e["D"],
        alpha=0.85
    )

    pi_aprox = metodo_svd_truncada(
        datos_e["M_densa"],
        k=50,
        alpha=0.85
    )

    cambios = analizar_nodos(
        G_e, pi_original, pi_aprox, 50
    )

    print("\n=== INCISO F: RENDIMIENTO ===")

    import matplotlib.pyplot as plt

    tamanos_f = [100, 500, 1000, 2000, 5000]

    tiempos = {
        "Potencia": [],
        "Pinv": [],
        "SVD truncada": [],
        "SVD dispersa": []
    }

    memorias = {
        "Potencia": [],
        "Pinv": [],
        "SVD truncada": [],
        "SVD dispersa": []
    }

    for n in tamanos_f:

        _, datos = grafos[f"BA_n{n}_conD"]

        # Método de la potencia
        _, t, m = medir_tiempo_memoria(
            metodo_potencia,
            datos["S_base"],
            datos["D"],
            0.85
        )

        tiempos["Potencia"].append(t)
        memorias["Potencia"].append(m)

        # SVD dispersa
        _, t, m = medir_tiempo_memoria(
            metodo_svds_disperso,
            datos["M_linop"],
            50,
            0.85
        )

        tiempos["SVD dispersa"].append(t)
        memorias["SVD dispersa"].append(m)

        # Métodos densos: limitar a n <= 1000
        if n <= 1000:

            _, t, m = medir_tiempo_memoria(
                metodo_pinv,
                datos["M_densa"],
                0.85
            )

            tiempos["Pinv"].append(t)
            memorias["Pinv"].append(m)

            _, t, m = medir_tiempo_memoria(
                metodo_svd_truncada,
                datos["M_densa"],
                50,
                0.85
            )

            tiempos["SVD truncada"].append(t)
            memorias["SVD truncada"].append(m)

        else:
            tiempos["Pinv"].append(np.nan)
            memorias["Pinv"].append(np.nan)

            tiempos["SVD truncada"].append(np.nan)
            memorias["SVD truncada"].append(np.nan)

        print(f"n={n} completado")

    # Grafica de tiempo
    plt.figure(figsize=(9, 6))

    for nombre, valores in tiempos.items():
        plt.plot(tamanos_f, valores, marker="o", label=nombre)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Numero de nodos")
    plt.ylabel("Tiempo (segundos)")
    plt.title("Tiempo de ejecucion")
    plt.legend()
    plt.grid()
    plt.savefig("inciso_f_tiempo.png", dpi=300)
    plt.show()

    # Grafica de memoria
    plt.figure(figsize=(9, 6))

    for nombre, valores in memorias.items():
        plt.plot(tamanos_f, valores, marker="o", label=nombre)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Numero de nodos")
    plt.ylabel("Memoria (MB)")
    plt.title("Consumo de memoria")
    plt.legend()
    plt.grid()
    plt.savefig("inciso_f_memoria.png", dpi=300)
    plt.show()

    print("\n=== INCISO G: ROBUSTEZ ===")

    semillas = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    errores = []
    taus = []

    for semilla in semillas:

        # Generar grafo original
        G_original = generar_grafo_sintetico(
            "BA", 500, con_dangling=True, seed=semilla
        )

        # Perturbar el 1% de sus aristas
        G_modificado = perturbar_grafo(
            G_original, porcentaje=0.01, seed=semilla + 1000
        )

        # Construir matrices dispersas
        datos_original = construir_matrices(
            G_original, alpha=0.85, construir_densa=False
        )

        datos_modificados = construir_matrices(
            G_modificado, alpha=0.85, construir_densa=False
        )

        # Calcular rankings
        pi_original = metodo_potencia(
            datos_original["S_base"],
            datos_original["D"],
            alpha=0.85
        )

        pi_modificado = metodo_potencia(
            datos_modificados["S_base"],
            datos_modificados["D"],
            alpha=0.85
        )

        # Medir diferencias
        error = np.linalg.norm(
            pi_original - pi_modificado, 1
        )

        tau, _ = kendalltau(
            pi_original, pi_modificado
        )

        errores.append(error)
        taus.append(tau)

        print(
            f"Semilla={semilla} | "
            f"Error L1={error:.6f} | "
            f"Kendall Tau={tau:.5f}"
        )

    print("\nRESULTADOS GENERALES")

    print(
        f"Error L1: media={np.mean(errores):.6f} | "
        f"desviacion={np.std(errores, ddof=1):.6f}"
    )

    print(
        f"Kendall Tau: media={np.mean(taus):.6f} | "
        f"desviacion={np.std(taus, ddof=1):.6f}"
    )