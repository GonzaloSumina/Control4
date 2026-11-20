
import os
import time
import tracemalloc
import random
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

# =============================================================================
# CONFIGURACION
# =============================================================================
RUTA_STANFORD = os.path.join("datasets", "web-Stanford.txt")
ALPHAS = [0.5, 0.85, 0.95, 0.99, 0.999]
ALPHA_BASE = 0.85
SEMILLAS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
TAMANOS = [100, 500, 1000, 2000, 5000]
MAX_NODOS_STANFORD = 5000
DIRECTORIO_RESULTADOS = "resultados_control4"
os.makedirs(DIRECTORIO_RESULTADOS, exist_ok=True)

# =============================================================================
# INCISO A: GENERACION DE GRAFOS Y CARGA DE STANFORD
# =============================================================================
def generar_grafo_sintetico(tipo, n, con_dangling=False, seed=42):
    """Genera ER o BA dirigido, controlando si existen nodos sin salida."""
    rng = random.Random(seed)
    if tipo == "ER":
        # Probabilidad esperada de grado de salida cercano a 8.
        G0 = nx.gnp_random_graph(n, min(8.0 / max(n - 1, 1), 1.0),
                                 seed=seed, directed=True)
    elif tipo == "BA":
        m = min(4, max(1, n - 1))
        G0 = nx.barabasi_albert_graph(n, m, seed=seed)
        G0 = nx.DiGraph()
        G0.add_nodes_from(range(n))
        # Orientacion aleatoria de cada arista no dirigida.
        base = nx.barabasi_albert_graph(n, m, seed=seed)
        for u, v in base.edges():
            if rng.random() < 0.5:
                G0.add_edge(u, v)
            else:
                G0.add_edge(v, u)
    else:
        raise ValueError("tipo debe ser 'ER' o 'BA'")

    if con_dangling:
        # Se fuerza que aproximadamente el 5% de nodos no tenga salida.
        cantidad = max(1, int(round(0.05 * n)))
        elegidos = rng.sample(list(G0.nodes()), cantidad)
        G0.remove_edges_from([(u, v) for u in elegidos for v in list(G0.successors(u))])
    else:
        # Ciclo dirigido: garantiza al menos una arista saliente por nodo.
        G0.add_edges_from((i, (i + 1) % n) for i in range(n))

    return nx.convert_node_labels_to_integers(G0)


def cargar_stanford(ruta=RUTA_STANFORD):
    if not os.path.exists(ruta):
        print(f"[AVISO] No se encontro {ruta}. Se ejecutaran ER y BA; Stanford se omitira.")
        return None
    print(f"Cargando Stanford desde: {ruta}")
    G = nx.read_edgelist(ruta, comments="#", create_using=nx.DiGraph(),
                         nodetype=int, data=False)
    G.remove_edges_from(nx.selfloop_edges(G))
    print(f"Stanford: {G.number_of_nodes():,} nodos, {G.number_of_edges():,} aristas")
    return G


def crear_subgrafo_bfs(G, n, seed=42):
    """Subgrafo inducido de la componente debil mayor; muestra local por BFS."""
    n = min(n, G.number_of_nodes())
    componente = max(nx.weakly_connected_components(G), key=len)
    H = G.subgraph(componente)
    inicio = max(H.degree, key=lambda par: par[1])[0]
    arbol = nx.bfs_tree(H.to_undirected(), source=inicio)
    seleccion = list(arbol.nodes())[:n]
    if len(seleccion) < n:
        usados = set(seleccion)
        seleccion.extend([u for u in H.nodes() if u not in usados][:n-len(seleccion)])
    return nx.convert_node_labels_to_integers(H.subgraph(seleccion).copy())


def construir_matrices(G, alpha=ALPHA_BASE, construir_densa=False):
    """
    A[i,j]=1 si j->i. S tiene columnas estocasticas; las columnas dangling
    se sustituyen por 1/n. Se evita np.outer (matriz densa temporal).
    """
    nodos = list(G.nodes())
    n = len(nodos)
    A = nx.to_scipy_sparse_array(G, nodelist=nodos, format="csc",
                                 dtype=float).T.tocsc()
    grados = np.asarray(A.sum(axis=0)).ravel()
    D = (grados == 0)
    inv = np.zeros(n)
    inv[~D] = 1.0 / grados[~D]
    S_base = (A @ sp.diags(inv, format="csc")).tocsc()

    # Construccion dispersa de las columnas dangling.
    dangling_cols = np.flatnonzero(D)
    if len(dangling_cols):
        filas = np.tile(np.arange(n, dtype=np.int32), len(dangling_cols))
        cols = np.repeat(dangling_cols.astype(np.int32), n)
        vals = np.full(n * len(dangling_cols), 1.0 / n)
        correccion = sp.csc_matrix((vals, (filas, cols)), shape=(n, n))
        S = (S_base + correccion).tocsc()
    else:
        S = S_base.copy()

    M = (sp.eye(n, format="csc") - alpha * S).tocsc()

    def matvec_M(v):
        masa = np.sum(v[D]) / n
        return v - alpha * (S_base @ v + masa * np.ones(n))

    def rmatvec_M(v):
        r = v - alpha * (S_base.T @ v)
        if np.any(D):
            r[D] -= alpha * np.sum(v) / n
        return r

    M_linop = spla.LinearOperator((n, n), matvec=matvec_M,
                                  rmatvec=rmatvec_M, dtype=float)
    return {
        "n": n, "D": D, "S_base": S_base, "S": S, "M": M,
        "M_linop": M_linop,
        "S_densa": S.toarray() if construir_densa else None,
        "M_densa": M.toarray() if construir_densa else None
    }


def construir_conjunto_grafos(G_stanford):
    grafos = {}
    print("\n=== INCISO A: GENERACION DE GRAFOS Y MATRICES ===")
    for tipo in ("ER", "BA"):
        for n in TAMANOS:
            for con_d in (True, False):
                nombre = f"{tipo}_n{n}_{'conD' if con_d else 'sinD'}"
                G = generar_grafo_sintetico(tipo, n, con_dangling=con_d,
                                            seed=1000+n)
                datos = construir_matrices(G, ALPHA_BASE,
                                           construir_densa=(n <= 1000))
                grafos[nombre] = (G, datos)
                print(f"[OK] {nombre:18s} | Dangling: {np.sum(datos['D']):5d} "
                      f"| S_disp nnz: {datos['S'].nnz:10d}")

    if G_stanford is not None:
        G_max = crear_subgrafo_bfs(G_stanford, MAX_NODOS_STANFORD)
        for n in TAMANOS:
            n_real = min(n, G_max.number_of_nodes())
            G = nx.convert_node_labels_to_integers(
                G_max.subgraph(range(n_real)).copy())
            datos = construir_matrices(G, ALPHA_BASE,
                                       construir_densa=(n_real <= 1000))
            grafos[f"Stanford_n{n_real}"] = (G, datos)
            # Se conserva una linea de salida clara para cada tamano Stanford.
            print(f"[A] Stanford n={G.number_of_nodes():5d} | "
                  f"aristas={G.number_of_edges():7d} | "
                  f"dangling={np.sum(datos['D']):5d}")
    return grafos


# =============================================================================
# INCISO B: CUATRO METODOS DE PAGERANK
# =============================================================================
def normalizar_l1(x):
    norma = np.linalg.norm(x, 1)
    if norma <= 1e-15:
        raise ArithmeticError("No se puede normalizar un vector de norma L1 casi cero.")
    return x / norma


def metodo_potencia(S_base, D, alpha=ALPHA_BASE, tol=1e-12, max_iter=200000):
    n = S_base.shape[0]
    pi = np.full(n, 1.0/n)
    for _ in range(max_iter):
        masa_dangling = np.sum(pi[D]) / n
        nuevo = alpha * (S_base @ pi + masa_dangling) + (1-alpha)/n
        if np.linalg.norm(nuevo-pi, 1) < tol:
            return normalizar_l1(nuevo)
        pi = nuevo
    raise RuntimeError("El metodo de potencia no convergio; aumenta max_iter.")


def metodo_pinv(M_densa, alpha=ALPHA_BASE):
    n = M_densa.shape[0]
    b = np.full(n, (1-alpha)/n)
    x = np.linalg.pinv(M_densa) @ b
    return normalizar_l1(x)


def calcular_svd(M_densa):
    # numpy entrega los valores singulares en orden descendente.
    return np.linalg.svd(M_densa, full_matrices=False)


def metodo_svd_truncada(M_densa, k, alpha=ALPHA_BASE, svd_cache=None):
    """Pseudoinversa truncada convencional: conserva los k mayores sigma."""
    n = M_densa.shape[0]
    k = max(1, min(int(k), n))
    b = np.full(n, (1-alpha)/n)
    U, s, Vt = calcular_svd(M_densa) if svd_cache is None else svd_cache
    x = Vt[:k, :].T @ ((U[:, :k].T @ b) / s[:k])
    return normalizar_l1(x)


def metodo_svds_disperso(M_linop, k, alpha=ALPHA_BASE):
    n = M_linop.shape[0]
    k = max(1, min(int(k), n-1))
    b = np.full(n, (1-alpha)/n)
    # LM corresponde a los mayores valores singulares, coherente con SVD truncada.
    u, s, vt = spla.svds(M_linop, k=k, which="LM",
                         return_singular_vectors=True,
                         tol=1e-8, maxiter=max(1000, 10*n))
    orden = np.argsort(s)[::-1]
    x = vt[orden, :].T @ ((u[:, orden].T @ b) / s[orden])
    return normalizar_l1(x)


# =============================================================================
# METRICAS Y MEDICION
# =============================================================================
def precision_at_m(pi_ref, pi_aprox, m):
    m = min(int(m), len(pi_ref))
    a = set(np.argsort(pi_ref)[::-1][:m])
    b = set(np.argsort(pi_aprox)[::-1][:m])
    return len(a & b) / m if m else np.nan


def comparar_rankings(ref, aprox):
    error = np.linalg.norm(ref-aprox, 1)
    tau, _ = kendalltau(ref, aprox)
    return float(error), float(tau)


def medir_tiempo_memoria(func, *args, repeticiones=5, **kwargs):
    tiempos, picos = [], []
    resultado = None
    for _ in range(repeticiones):
        tracemalloc.start()
        t0 = time.perf_counter()
        resultado = func(*args, **kwargs)
        tiempos.append(time.perf_counter()-t0)
        _, pico = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        picos.append(pico)
    return resultado, float(np.mean(tiempos)), float(max(picos)/(1024**2))


# =============================================================================
# INCISO B: VALIDACION FORMAL
# =============================================================================
def inciso_b_validacion(grafos):
    print("\n=== INCISO B: VALIDACION (TOLERANCIA 1e-8) ===")
    candidatos = ["ER_n100_conD", "BA_n100_conD"]
    if "Stanford_n100" in grafos:
        candidatos.append("Stanford_n100")
    for nombre in candidatos:
        G, d = grafos[nombre]
        pi_pot = metodo_potencia(d["S_base"], d["D"])
        pi_pinv = metodo_pinv(d["M_densa"])
        pi_svd = metodo_svd_truncada(d["M_densa"], d["n"])
        e_pinv = np.linalg.norm(pi_pot-pi_pinv, 1)
        e_svd = np.linalg.norm(pi_pot-pi_svd, 1)
        print(f"{nombre:20s} | potencia/pinv={e_pinv:.3e} "
              f"| potencia/SVD(k=n)={e_svd:.3e}")
        assert e_pinv < 1e-8 and e_svd < 1e-8, (
            f"Validacion fallida para {nombre}: revisar matrices/metodos.")
    print("Validacion superada: potencia, pinv y SVD con k=n coinciden (<1e-8).")


# =============================================================================
# INCISO C: ESPECTRO Y CONDICIONAMIENTO
# =============================================================================
def inciso_c_espectro(grafos):
    print("\n=== INCISO C: ESPECTRO Y CONDICIONAMIENTO ===")
    nombre = "BA_n500_conD"
    G, _ = grafos[nombre]
    resumen = []
    espectros = {}
    for alpha in ALPHAS:
        d = construir_matrices(G, alpha, construir_densa=True)
        s = np.linalg.svd(d["M_densa"], compute_uv=False)
        kappa = s[0]/s[-1] if s[-1] > 0 else np.inf
        r1 = int(np.sum(s > 1e-2*s[0]))
        r2 = int(np.sum(s > 1e-4*s[0]))
        resumen.append((alpha, s[-1], kappa, r1, r2))
        espectros[alpha] = s
        print(f"alpha={alpha:5.3f} | sigma_min={s[-1]:.6e} "
              f"| kappa2={kappa:.6e} | rango(1e-2)={r1} | rango(1e-4)={r2}")

    fig, ax = plt.subplots(figsize=(9, 6))
    for alpha, s in espectros.items():
        ax.semilogy(np.arange(1, len(s)+1), s, label=f"alpha={alpha}")
    ax.set_xlabel("Indice singular i")
    ax.set_ylabel("Valor singular sigma_i (escala log)")
    ax.set_title("Espectro singular de M_alpha (BA, n=500, con dangling)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(DIRECTORIO_RESULTADOS, "inciso_c_espectro.png"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([x[0] for x in resumen], [x[2] for x in resumen], marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Numero de condicion kappa_2(M_alpha)")
    ax.set_title("Condicionamiento frente a alpha")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DIRECTORIO_RESULTADOS, "inciso_c_condicionamiento.png"), dpi=220)
    plt.close(fig)
    return resumen


# =============================================================================
# INCISO D: ERROR, RANKING Y k MINIMO
# =============================================================================
def inciso_d_calidad(grafos):
    print("\n=== INCISO D: ERROR Y CALIDAD DEL RANKING ===")
    # Comparar ER, BA y Stanford para observar distribuciones de grado distintas.
    conjuntos = ["ER_n100_conD", "ER_n500_conD",
                 "BA_n100_conD", "BA_n500_conD"]
    if "Stanford_n100" in grafos:
        conjuntos += ["Stanford_n100", "Stanford_n500"]
    resultados = []
    ks_por_n = {
        100: [5, 10, 20, 40, 60, 80, 100],
        500: [5, 10, 25, 50, 100, 150, 250, 400, 500]
    }

    for nombre in conjuntos:
        G, _ = grafos[nombre]
        n = len(G)
        ks = ks_por_n[n]
        for alpha in ALPHAS:
            d = construir_matrices(G, alpha, construir_densa=True)
            ref = metodo_potencia(d["S_base"], d["D"], alpha)
            cache = calcular_svd(d["M_densa"])
            k_min = None
            print(f"\n[{nombre}] alpha={alpha}")
            for k in ks:
                aprox = metodo_svd_truncada(d["M_densa"], k, alpha, cache)
                err, tau = comparar_rankings(ref, aprox)
                p10 = precision_at_m(ref, aprox, 10)
                p50 = precision_at_m(ref, aprox, 50)
                p100 = precision_at_m(ref, aprox, 100)
                print(f"k={k:3d} | L1={err:.6e} | tau={tau:.5f} "
                      f"| P@10={p10:.3f} | P@50={p50:.3f} | P@100={p100:.3f}")
                resultados.append((nombre, n, alpha, k, err, tau, p10, p50, p100))
                if k_min is None and np.isfinite(tau) and tau > 0.95:
                    k_min = k
            print(f"k minimo con Kendall tau > 0.95: {k_min}")

            # El metodo disperso se compara tambien en varios k.
            for k in sorted(set([min(10, n-1), min(50, n-1), min(100, n-1)])):
                try:
                    aprox_s = metodo_svds_disperso(d["M_linop"], k, alpha)
                    err_s, tau_s = comparar_rankings(ref, aprox_s)
                    print(f"SVD dispersa k={k:3d} | L1={err_s:.6e} | tau={tau_s:.5f}")
                except (spla.ArpackNoConvergence, RuntimeError, ValueError) as exc:
                    print(f"[AVISO] SVD dispersa k={k} no convergio: {exc}")

    # CSV reproducible para analizar resultados fuera del programa.
    salida = os.path.join(DIRECTORIO_RESULTADOS, "inciso_d_resultados.csv")
    np.savetxt(salida, np.array([[r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]]
                                 for r in resultados]),
               delimiter=",", header="n,alpha,k,error_L1,kendall_tau,P10,P50,P100",
               comments="")
    print(f"Resultados numericos guardados en {salida}")
    return resultados


# =============================================================================
# INCISO E: ANALISIS POR NODO Y ULTIMOS VECTORES SINGULARES
# =============================================================================
def analizar_nodos(G, datos, pi_ref, pi_aprox, k=10, etiqueta="grafo"):
    n = len(pi_ref)
    rango_ref = np.argsort(np.argsort(-pi_ref))
    rango_aprox = np.argsort(np.argsort(-pi_aprox))
    cambio_pos = np.abs(rango_ref-rango_aprox)
    cambio_pi = np.abs(pi_ref-pi_aprox)

    # Los ultimos vectores de V corresponden a los menores valores singulares.
    U, s, Vt = calcular_svd(datos["M_densa"])
    V_pequenos = Vt[-k:, :]
    participacion = np.sum(np.abs(V_pequenos), axis=0)

    entrada = np.array([G.in_degree(v) for v in range(n)])
    salida = np.array([G.out_degree(v) for v in range(n)])
    bet = nx.betweenness_centrality(G, normalized=True)
    bet = np.array([bet[v] for v in range(n)])

    print("\n=== INCISO E: NODOS MAS AFECTADOS ===")
    print("Nodo | pi referencia | error abs | cambio ranking | in | out | participacion | betweenness | tipo")
    afectados = np.argsort(cambio_pos)[::-1][:20]
    for v in afectados:
        if salida[v] == 0:
            tipo = "dangling"
        elif salida[v] <= np.percentile(salida, 25):
            tipo = "bajo grado"
        elif bet[v] >= np.percentile(bet, 90):
            tipo = "puente"
        elif entrada[v] >= np.percentile(entrada, 90):
            tipo = "hub"
        else:
            tipo = "intermedio"
        print(f"{v:4d} | {pi_ref[v]:.6e} | {cambio_pi[v]:.3e} | "
              f"{cambio_pos[v]:5d} | {entrada[v]:4d} | {salida[v]:4d} | "
              f"{participacion[v]:.5f} | {bet[v]:.5f} | {tipo}")

    # Grafo completo: tamano proporcional al PageRank, color = cambio de puesto.
    pos = nx.spring_layout(G, seed=42, iterations=80)
    tamanos = 30 + 1800*pi_ref/(pi_ref.max() + 1e-15)
    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.05, width=0.25,
                           edge_color="gray", arrows=False)
    nodos = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=tamanos,
                                   node_color=cambio_pos, cmap="coolwarm")
    fig.colorbar(nodos, ax=ax, label="Cambio de posicion en el ranking")
    ax.set_title(f"{etiqueta}: tamano proporcional a PageRank; color = cambio de ranking")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(DIRECTORIO_RESULTADOS, f"inciso_e_{etiqueta}.png"), dpi=220)
    plt.close(fig)

    # Exportar tabla completa por nodo.
    tabla = np.column_stack((np.arange(n), pi_ref, pi_aprox, cambio_pi,
                             cambio_pos, entrada, salida, participacion, bet))
    np.savetxt(os.path.join(DIRECTORIO_RESULTADOS, f"inciso_e_nodos_{etiqueta}.csv"),
               tabla, delimiter=",",
               header="nodo,PageRank_ref,PageRank_aprox,error_abs,cambio_ranking,grado_entrada,grado_salida,participacion_singular,betweenness",
               comments="")
    print("Menores valores singulares considerados:", s[-k:])
    print("Tabla por nodo y grafico guardados en la carpeta de resultados.")
    return cambio_pos, cambio_pi, participacion


# =============================================================================
# INCISO F: RENDIMIENTO, MEMORIA Y ORDEN EMPIRICO
# =============================================================================
def pendiente_loglog(ns, valores):
    ns = np.asarray(ns, dtype=float)
    valores = np.asarray(valores, dtype=float)
    mascara = np.isfinite(valores) & (valores > 0)
    if np.sum(mascara) < 2:
        return np.nan
    return float(np.polyfit(np.log(ns[mascara]), np.log(valores[mascara]), 1)[0])


def inciso_f_rendimiento(grafos):
    print("\n=== INCISO F: RENDIMIENTO (PROMEDIO DE 5 CORRIDAS) ===")
    metodos = ["Potencia", "Pinv", "SVD truncada", "SVD dispersa"]
    tiempos = {m: [] for m in metodos}
    memorias = {m: [] for m in metodos}
    ns_validos = []
    for n in TAMANOS:
        nombre = f"BA_n{n}_conD"
        G, d = grafos[nombre]
        ns_validos.append(n)

        _, t, mem = medir_tiempo_memoria(metodo_potencia, d["S_base"], d["D"], ALPHA_BASE)
        tiempos["Potencia"].append(t); memorias["Potencia"].append(mem)

        try:
            _, t, mem = medir_tiempo_memoria(metodo_svds_disperso, d["M_linop"],
                                             min(20, n-1), ALPHA_BASE)
        except Exception as exc:
            t, mem = np.nan, np.nan
            print(f"[AVISO] SVD dispersa n={n}: {exc}")
        tiempos["SVD dispersa"].append(t); memorias["SVD dispersa"].append(mem)

        if n <= 500:
            # Pinv incluye la descomposicion/inversion densa.
            _, t, mem = medir_tiempo_memoria(metodo_pinv, d["M_densa"], ALPHA_BASE)
            tiempos["Pinv"].append(t); memorias["Pinv"].append(mem)
            _, t, mem = medir_tiempo_memoria(metodo_svd_truncada, d["M_densa"],
                                             min(50, n), ALPHA_BASE)
            tiempos["SVD truncada"].append(t); memorias["SVD truncada"].append(mem)
        else:
            tiempos["Pinv"].append(np.nan); memorias["Pinv"].append(np.nan)
            tiempos["SVD truncada"].append(np.nan); memorias["SVD truncada"].append(np.nan)
        print(f"n={n:5d}: mediciones completadas")

    for titulo, datos_plot, ylabel, archivo in [
        ("Tiempo de ejecucion", tiempos, "Tiempo (s)", "inciso_f_tiempo.png"),
        ("Memoria medida", memorias, "Memoria (MB)", "inciso_f_memoria.png")
    ]:
        fig, ax = plt.subplots(figsize=(9, 6))
        for nombre, vals in datos_plot.items():
            ax.plot(ns_validos, vals, marker="o", label=nombre)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Numero de nodos n"); ax.set_ylabel(ylabel)
        ax.set_title(titulo); ax.grid(True, which="both", alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(DIRECTORIO_RESULTADOS, archivo), dpi=220)
        plt.close(fig)

    print("\nPendientes empiricas en escala log-log (tiempo):")
    for m in metodos:
        p = pendiente_loglog(ns_validos, tiempos[m])
        print(f"{m:16s}: exponente aproximado = {p:.3f}")
    print("Costos teoricos orientativos:")
    print("- Potencia: O(I*(n + |E|)) por iteraciones, usando matriz dispersa.")
    print("- Pinv/SVD densa: O(n^3) tiempo y O(n^2) memoria.")
    print("- SVD dispersa: depende de k, iteraciones y nnz; evita almacenar toda la matriz densa.")
    print("[NOTA] tracemalloc puede no contar toda la memoria nativa de NumPy/SciPy.")


# =============================================================================
# INCISO G: ROBUSTEZ, 10 SEMILLAS Y PERTURBACION DEL 1% DE ARISTAS
# =============================================================================
def perturbar_grafo(G, porcentaje=0.01, seed=42):
    rng = random.Random(seed)
    H = G.copy()
    aristas = list(H.edges())
    cantidad = max(1, int(round(len(aristas)*porcentaje)))
    eliminadas = rng.sample(aristas, min(cantidad, len(aristas)))
    H.remove_edges_from(eliminadas)

    nodos = list(H.nodes())
    existentes = set(H.edges())
    agregadas = 0
    max_intentos = max(2000, cantidad*200)
    for _ in range(max_intentos):
        if agregadas >= len(eliminadas):
            break
        u, v = rng.choice(nodos), rng.choice(nodos)
        if u != v and (u, v) not in existentes:
            H.add_edge(u, v); existentes.add((u, v)); agregadas += 1
    if agregadas != len(eliminadas):
        warnings.warn(f"Se agregaron {agregadas}/{len(eliminadas)} aristas nuevas.")
    return H


def inciso_g_robustez(grafos):
    print("\n=== INCISO G: ROBUSTEZ ANTE PERTURBACIONES DEL 1% ===")
    nombre = "Stanford_n2000" if "Stanford_n2000" in grafos else "BA_n2000_conD"
    G, d = grafos[nombre]
    referencia = metodo_potencia(d["S_base"], d["D"], ALPHA_BASE)
    errores, taus = [], []
    for seed in SEMILLAS:
        Gp = perturbar_grafo(G, 0.01, seed)
        dp = construir_matrices(Gp, ALPHA_BASE, construir_densa=False)
        pip = metodo_potencia(dp["S_base"], dp["D"], ALPHA_BASE)
        err, tau = comparar_rankings(referencia, pip)
        errores.append(err); taus.append(tau)
        print(f"Semilla={seed:3d} | error L1={err:.6e} | Kendall tau={tau:.6f}")
    print(f"Error L1: media={np.mean(errores):.6e}, desviacion={np.std(errores, ddof=1):.6e}")
    print(f"Tau:      media={np.mean(taus):.6f}, desviacion={np.std(taus, ddof=1):.6f}")
    np.savetxt(os.path.join(DIRECTORIO_RESULTADOS, "inciso_g_robustez.csv"),
               np.column_stack((SEMILLAS, errores, taus)), delimiter=",",
               header="semilla,error_L1,kendall_tau", comments="")

# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    G_stanford = cargar_stanford()
    grafos = construir_conjunto_grafos(G_stanford)

    inciso_b_validacion(grafos)
    resumen_c = inciso_c_espectro(grafos)
    resultados_d = inciso_d_calidad(grafos)

    print("\n=== INCISO E: ANALISIS POR NODO ===")
    nombre_e = "Stanford_n500" if "Stanford_n500" in grafos else "BA_n500_conD"
    G_e, d_e = grafos[nombre_e]
    pi_ref = metodo_potencia(d_e["S_base"], d_e["D"], ALPHA_BASE)
    pi_aprox = metodo_svd_truncada(d_e["M_densa"], min(50, len(G_e)), ALPHA_BASE)
    analizar_nodos(G_e, d_e, pi_ref, pi_aprox, k=min(10, len(G_e)),
                   etiqueta=nombre_e.lower())

    inciso_f_rendimiento(grafos)
    inciso_g_robustez(grafos)

    print(f"\nEjecucion terminada. Resultados guardados en: {DIRECTORIO_RESULTADOS}/")
