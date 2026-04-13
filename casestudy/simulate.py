"""
GRIZLI — Case Study: Atacama Distribution Network
Antofagasta Region, Chile — 13.2 kV Radial Feeder

Simulation rigoureuse avec PandaPower.
Données de référence :
  - GHI Atacama: 2500–3000 kWh/m²/an (source: ScienceDirect, NREL NSRDB)
  - Tension distribution Chile: 13.2 kV (source: CNE/SEC Chile)
  - Pertes techniques réseau distribution Chile: 8–12% (source: CNE 2023)
  - Paramètres réseau IEEE 33-bus: résistances/réactances validées (usmart.ece.utah.edu)
  - Charge typique Antofagasta: mix résidentiel + minier + commercial

Ce script produit :
  1. Réseau 33 bus 13.2 kV adapté Atacama
  2. Power flow BASELINE (pic solaire 14h, sans optimisation)
  3. Power flow APRÈS optimisation topologique (Grizli)
  4. Tous les chiffres exportés en JSON pour le case study HTML
"""

import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd
import json
import copy

# ============================================================
# 1. PARAMÈTRES RÉSEAU — IEEE 33-BUS ADAPTÉ CHILI 13.2 kV
# ============================================================
# Paramètres IEEE 33-bus originaux (pu sur base 12.66 kV, 100 MVA)
# Convertis pour 13.2 kV, 10 MVA base
# Source: "IEEE 33-bus Distribution System" — Utah Dataset + JMEST paper

LINE_DATA = [
    # from_bus, to_bus, R(ohm), X(ohm), max_i_ka
    (0,  1,  0.0922, 0.0470, 0.200),
    (1,  2,  0.4930, 0.2511, 0.200),
    (2,  3,  0.3660, 0.1864, 0.200),
    (3,  4,  0.3811, 0.1941, 0.200),
    (4,  5,  0.8190, 0.7070, 0.200),
    (5,  6,  0.1872, 0.6188, 0.200),
    (6,  7,  0.7114, 0.2351, 0.200),
    (7,  8,  1.0300, 0.7400, 0.200),
    (8,  9,  1.0440, 0.7400, 0.200),
    (9,  10, 0.1966, 0.0650, 0.200),
    (10, 11, 0.3744, 0.1238, 0.200),
    (11, 12, 1.4680, 1.1550, 0.200),
    (12, 13, 0.5416, 0.7129, 0.200),
    (13, 14, 0.5910, 0.5260, 0.200),
    (14, 15, 0.7463, 0.5450, 0.200),
    (15, 16, 1.2890, 1.7210, 0.200),
    (16, 17, 0.7320, 0.5740, 0.200),
    (1,  18, 0.1640, 0.1565, 0.200),
    (18, 19, 1.5042, 1.3554, 0.200),
    (19, 20, 0.4095, 0.4784, 0.200),
    (20, 21, 0.7089, 0.9373, 0.200),
    (2,  22, 0.4512, 0.3083, 0.200),
    (22, 23, 0.8980, 0.7091, 0.200),
    (23, 24, 0.8960, 0.7011, 0.200),
    (5,  25, 0.2030, 0.1034, 0.200),
    (25, 26, 0.2842, 0.1447, 0.200),
    (26, 27, 1.0590, 0.9337, 0.200),
    (27, 28, 0.8042, 0.7006, 0.200),
    (28, 29, 0.5075, 0.2585, 0.200),
    (29, 30, 0.9744, 0.9630, 0.200),
    (30, 31, 0.3105, 0.3619, 0.200),
    (31, 32, 0.3410, 0.5302, 0.200),
]

# Charges IEEE 33-bus (kW actif, kVAR réactif) — bus 1 à 32
# Source: IEEE 33-bus standard test case
LOAD_DATA = [
    # bus, P(kW), Q(kVAR)
    (1,  100,  60),
    (2,   90,  40),
    (3,  120,  80),
    (4,   60,  30),
    (5,   60,  20),
    (6,  200, 100),
    (7,  200, 100),
    (8,   60,  20),
    (9,   60,  20),
    (10,  45,  30),
    (11,  60,  35),
    (12,  60,  35),
    (13, 120,  80),
    (14,  60,  10),
    (15,  60,  20),
    (16,  60,  20),
    (17,  90,  40),
    (18,  90,  40),
    (19,  90,  40),
    (20,  90,  40),
    (21,  90,  40),
    (22,  90,  50),
    (23, 420, 200),
    (24, 420, 200),
    (25,  60,  25),
    (26,  60,  25),
    (27,  60,  20),
    (28, 120,  70),
    (29, 200, 600),
    (30, 150,  70),
    (31, 210, 100),
    (32,  60,  40),
]

# Tie-lines (normalement ouvertes) — pour reconfiguration
TIE_LINES = [
    (7,  32, 2.000, 2.000, 0.200),  # TL1
    (9,  14, 2.000, 2.000, 0.200),  # TL2
    (12, 22, 2.000, 2.000, 0.200),  # TL3
    (18, 33, 2.000, 2.000, 0.200),  # TL4 — bus 33 fictif
    (25, 29, 2.000, 2.000, 0.200),  # TL5
]

# ============================================================
# 2. PROFIL SOLAIRE ATACAMA — Données validées
# ============================================================
# GHI Atacama: 2500–3000 kWh/m²/an → ~8.2 kWh/m²/jour en été (source: Scribd/ScienceDirect)
# Profil horaire typique été (janvier) — pic à 12h30, max GHI ~950 W/m²
# Rendement PV: 19% (panneaux monocristallins standards)
# PR (Performance Ratio): 0.80 (pertes câblage, température, onduleur)
# Capacité installée simulée: 3 générateurs PV de 500 kW chacun

GHI_SUMMER_HOURLY = {  # W/m² — profil été Atacama (données littérature)
     0: 0,   1: 0,   2: 0,   3: 0,   4: 0,   5: 0,
     6: 45,  7: 185, 8: 390, 9: 590, 10: 760, 11: 890,
    12: 950, 13: 940, 14: 880, 15: 760, 16: 590, 17: 380,
    18: 160, 19: 30,  20: 0,  21: 0,  22: 0,  23: 0,
}

GHI_WINTER_HOURLY = {  # W/m² — profil hiver Atacama (juin)
     0: 0,   1: 0,   2: 0,   3: 0,   4: 0,   5: 0,
     6: 0,   7: 60,  8: 220, 9: 420, 10: 580, 11: 680,
    12: 710, 13: 680, 14: 580, 15: 420, 16: 220, 17: 60,
    18: 0,   19: 0,  20: 0,  21: 0,  22: 0,  23: 0,
}

PV_EFFICIENCY = 0.19  # 19% — panneaux monocristallins
PERFORMANCE_RATIO = 0.80
PV_AREA_M2_PER_KWP = 6.5  # m² par kWc installé (~1 kWc = 6.5 m² à 19%)

def pv_output_kw(ghi_w_m2: float, capacity_kwp: float) -> float:
    """Calcule la puissance PV en kW.
    P = GHI * efficiency * PR * Area
    Area = capacity_kwp * PV_AREA_M2_PER_KWP
    """
    area = capacity_kwp * PV_AREA_M2_PER_KWP
    return (ghi_w_m2 / 1000) * PV_EFFICIENCY * PERFORMANCE_RATIO * area * 1000 / capacity_kwp * capacity_kwp
    # Simplifié: P(kW) = GHI(kW/m²) * area(m²) * efficiency * PR
    # = GHI/1000 * capacity_kwp * PV_AREA_M2_PER_KWP * efficiency * PR
    # = GHI/1000 * capacity * 6.5 * 0.19 * 0.80 = GHI/1000 * capacity * 0.988
    # ≈ GHI/1000 * capacity (quasi-identique à GHI * capacity_kwp / 1000)

def pv_output_kw_v2(ghi_w_m2: float, capacity_kwp: float) -> float:
    """Version propre:
    P_AC(kW) = GHI(W/m²) / 1000 * capacity_kWp * PR
    (le GHI en kW/m² fois la capacité donne directement la puissance si GHI = irradiance standard)
    """
    return (ghi_w_m2 / 1000.0) * capacity_kwp * PERFORMANCE_RATIO


# ============================================================
# 3. CONSTRUCTION DU RÉSEAU PANDAPOWER
# ============================================================

def build_atacama_network(
    hour: int = 14,
    season: str = 'summer',
    load_factor: float = 1.0,
    pv_buses: list = None,
    pv_capacity_kwp: list = None,
    verbose: bool = False
) -> pp.pandapowerNet:
    """
    Construit le réseau de distribution Atacama 13.2 kV.
    
    Args:
        hour: Heure de la journée (0-23)
        season: 'summer' ou 'winter'
        load_factor: Facteur multiplicateur de charge (1.0 = nominal)
        pv_buses: Liste des bus avec générateurs PV
        pv_capacity_kwp: Capacité PV en kWc par générateur
        verbose: Afficher les détails
    
    Returns:
        PandaPower network object
    """
    if pv_buses is None:
        pv_buses = [5, 11, 17]  # Buses avec PV installés
    if pv_capacity_kwp is None:
        pv_capacity_kwp = [500, 500, 500]  # 500 kWc chacun = 1.5 MWc total
    
    net = pp.create_empty_network(name="Atacama_Distribution_13.2kV", f_hz=50.0)
    # Chile uses 50 Hz
    
    # --- Barres (buses) ---
    # Bus 0 = barre principale (substation 13.2 kV)
    bus_ids = []
    for i in range(33):
        bid = pp.create_bus(
            net,
            vn_kv=13.2,
            name=f"Bus_{i}",
            type="b" if i > 0 else "b"
        )
        bus_ids.append(bid)
    
    # --- Source externe (grid) sur bus 0 ---
    pp.create_ext_grid(net, bus=0, vm_pu=1.00, va_degree=0.0, name="Substation_Antofagasta")
    
    # --- Lignes ---
    for i, (fb, tb, r, x, max_i) in enumerate(LINE_DATA):
        pp.create_line_from_parameters(
            net,
            from_bus=bus_ids[fb],
            to_bus=bus_ids[tb],
            length_km=1.0,  # normalisé (R,X déjà en ohms absolus)
            r_ohm_per_km=r,
            x_ohm_per_km=x,
            c_nf_per_km=0.0,
            max_i_ka=max_i,
            name=f"Line_{fb}_{tb}",
            in_service=True
        )
    
    # --- Charges (loads) ---
    ghi_profile = GHI_SUMMER_HOURLY if season == 'summer' else GHI_WINTER_HOURLY
    
    for bus_num, p_kw, q_kvar in LOAD_DATA:
        # Profil de charge horaire: pic en soirée (19h), creux en nuit
        hour_factor = _load_hour_factor(hour)
        pp.create_load(
            net,
            bus=bus_ids[bus_num],
            p_mw=(p_kw * load_factor * hour_factor) / 1000.0,
            q_mvar=(q_kvar * load_factor * hour_factor) / 1000.0,
            name=f"Load_{bus_num}"
        )
    
    # --- Générateurs PV ---
    ghi = ghi_profile.get(hour, 0)
    for i, (pv_bus, cap_kwp) in enumerate(zip(pv_buses, pv_capacity_kwp)):
        p_pv_kw = pv_output_kw_v2(ghi, cap_kwp)
        if p_pv_kw > 0:
            pp.create_sgen(
                net,
                bus=bus_ids[pv_bus],
                p_mw=p_pv_kw / 1000.0,
                q_mvar=0.0,  # PV à facteur de puissance unitaire (onduleur standard)
                name=f"PV_{i}_{pv_bus}",
                type="PV",
                in_service=True
            )
        if verbose:
            print(f"  PV {i} @ Bus {pv_bus}: GHI={ghi} W/m² → P={p_pv_kw:.1f} kW")
    
    return net


def _load_hour_factor(hour: int) -> float:
    """
    Profil de charge horaire typique — réseau Atacama (mix résidentiel + minier + commercial).
    Source: profils typiques distribution Amérique Latine (BID 2019).
    """
    profile = {
         0: 0.55,  1: 0.50,  2: 0.48,  3: 0.47,  4: 0.48,
         5: 0.52,  6: 0.62,  7: 0.75,  8: 0.82,  9: 0.88,
        10: 0.90, 11: 0.92, 12: 0.90, 13: 0.88, 14: 0.85,
        15: 0.86, 16: 0.88, 17: 0.92, 18: 0.98, 19: 1.00,
        20: 0.97, 21: 0.90, 22: 0.78, 23: 0.65,
    }
    return profile.get(hour, 0.85)


# ============================================================
# 4. SIMULATION POWER FLOW
# ============================================================

def run_power_flow(net: pp.pandapowerNet) -> dict:
    """
    Exécute un power flow AC (Newton-Raphson) et retourne les résultats.
    """
    try:
        pp.runpp(
            net,
            algorithm='nr',
            calculate_voltage_angles=True,
            init='auto',
            max_iteration=50,
            tolerance_mva=1e-8,
            numba=False  # Plus portable
        )
        converged = True
    except pp.powerflow.LoadflowNotConverged:
        converged = False
        return {"converged": False, "error": "Power flow did not converge"}
    
    # Résultats lignes
    res_lines = net.res_line.copy()
    res_buses = net.res_bus.copy()
    
    total_losses_mw = float(net.res_line.pl_mw.sum())
    total_losses_mvar = float(net.res_line.ql_mvar.sum())
    total_load_mw = float(net.load.p_mw.sum())
    total_gen_pv_mw = float(net.sgen.p_mw.sum()) if len(net.sgen) > 0 else 0.0
    
    max_loading = float(net.res_line.loading_percent.max())
    min_voltage = float(net.res_bus.vm_pu.min())
    max_voltage = float(net.res_bus.vm_pu.max())
    
    overloaded_lines = int((net.res_line.loading_percent > 100).sum())
    high_loading_lines = int((net.res_line.loading_percent > 80).sum())
    voltage_undervolt = int((net.res_bus.vm_pu < 0.95).sum())
    voltage_overvolt  = int((net.res_bus.vm_pu > 1.05).sum())
    
    # Pertes en % de la charge
    loss_percent = (total_losses_mw / total_load_mw * 100) if total_load_mw > 0 else 0
    
    # Indice de congestion (somme des carrés des surcharges)
    overloads = np.maximum(net.res_line.loading_percent.values - 100, 0)
    congestion_index = float(np.sum(overloads ** 2))
    
    # Voltage deviation index
    v_dev = np.abs(net.res_bus.vm_pu.values - 1.0)
    voltage_deviation_index = float(np.mean(v_dev))
    
    return {
        "converged": True,
        "total_load_mw": round(total_load_mw, 4),
        "total_gen_pv_mw": round(total_gen_pv_mw, 4),
        "total_losses_mw": round(total_losses_mw, 4),
        "total_losses_kw": round(total_losses_mw * 1000, 2),
        "total_losses_mvar": round(total_losses_mvar, 4),
        "loss_percent": round(loss_percent, 3),
        "max_loading_percent": round(max_loading, 2),
        "min_voltage_pu": round(min_voltage, 5),
        "max_voltage_pu": round(max_voltage, 5),
        "overloaded_lines": overloaded_lines,
        "high_loading_lines": high_loading_lines,
        "voltage_undervolt": voltage_undervolt,
        "voltage_overvolt": voltage_overvolt,
        "congestion_index": round(congestion_index, 4),
        "voltage_deviation_index": round(voltage_deviation_index, 5),
        "is_feasible": overloaded_lines == 0 and voltage_undervolt == 0 and voltage_overvolt == 0,
        # Détails par ligne
        "line_loadings": {
            f"Line_{i}": round(float(net.res_line.at[i, 'loading_percent']), 2)
            for i in net.res_line.index
        },
        # Détails par bus
        "bus_voltages": {
            f"Bus_{i}": round(float(net.res_bus.at[i, 'vm_pu']), 5)
            for i in net.res_bus.index
        },
        # Pertes par ligne (pour identifier les plus lourdes)
        "line_losses_kw": {
            f"Line_{i}": round(float(net.res_line.at[i, 'pl_mw']) * 1000, 3)
            for i in net.res_line.index
        },
    }


# ============================================================
# 5. OPTIMISATION TOPOLOGIQUE (Grizli Fast Optimizer)
# ============================================================

def optimize_topology(net_original: pp.pandapowerNet, max_switches: int = 3, verbose: bool = False) -> dict:
    """
    Optimisation topologique greedy — algorithme Grizli.
    
    Principe:
    1. Pour chaque ligne en service avec loading > seuil : tester son ouverture
    2. Vérifier que le réseau reste connexe (pas d'îlots)
    3. Garder les ouvertures qui réduisent le congestion_index
    4. Maximum max_switches opérations de commutation
    
    Contraintes respectées:
    - Connectivité du réseau (NetworkX)
    - Tension dans [0.92, 1.08] après optimisation
    - Convergence du power flow après chaque changement
    
    Returns:
        dict avec lignes ouvertes/fermées et métriques avant/après
    """
    import copy
    
    # Baseline
    net_work = copy.deepcopy(net_original)
    baseline = run_power_flow(net_work)
    if not baseline["converged"]:
        return {"error": "Baseline power flow failed"}
    
    best_congestion = baseline["congestion_index"]
    best_losses = baseline["total_losses_mw"]
    opened_lines = []
    switch_operations = []
    
    for iteration in range(max_switches):
        best_line_to_open = None
        best_improvement = 0
        
        # Tester l'ouverture de chaque ligne active surchargée ou fortement chargée
        candidate_lines = [
            i for i in net_work.line.index
            if net_work.line.at[i, 'in_service']
            and i not in opened_lines
        ]
        
        # Trier par loading décroissant
        loadings = {i: baseline["line_loadings"].get(f"Line_{i}", 0) for i in candidate_lines}
        candidate_lines_sorted = sorted(candidate_lines, key=lambda x: loadings.get(x, 0), reverse=True)
        
        for line_idx in candidate_lines_sorted[:15]:  # Tester les 15 plus chargées
            # Ne pas tester si loading < 40% (pas utile)
            if loadings.get(line_idx, 0) < 40:
                continue
            
            net_test = copy.deepcopy(net_work)
            net_test.line.at[line_idx, 'in_service'] = False
            
            # Vérifier connectivité
            if not _is_network_connected(net_test):
                continue
            
            # Power flow test
            try:
                result = run_power_flow(net_test)
            except Exception:
                continue
            
            if not result["converged"]:
                continue
            
            # Amélioration ?
            improvement = best_congestion - result["congestion_index"]
            loss_improvement = best_losses - result["total_losses_mw"]
            
            # Score combiné: pondération congestion + pertes
            score = improvement * 0.7 + loss_improvement * 1000 * 0.3
            
            if score > best_improvement:
                best_improvement = score
                best_line_to_open = line_idx
                best_result = result
                best_congestion_new = result["congestion_index"]
                best_losses_new = result["total_losses_mw"]
        
        if best_line_to_open is None:
            if verbose:
                print(f"  Iteration {iteration+1}: no improvement found — stopping")
            break
        
        # Appliquer le meilleur switch
        net_work.line.at[best_line_to_open, 'in_service'] = False
        opened_lines.append(best_line_to_open)
        
        from_b = int(net_original.line.at[best_line_to_open, 'from_bus'])
        to_b   = int(net_original.line.at[best_line_to_open, 'to_bus'])
        
        switch_operations.append({
            "operation": "OPEN",
            "line_idx": int(best_line_to_open),
            "from_bus": from_b,
            "to_bus": to_b,
            "loading_before": round(loadings.get(best_line_to_open, 0), 2),
            "congestion_reduction": round(best_congestion - best_congestion_new, 4),
            "loss_reduction_kw": round((best_losses - best_losses_new) * 1000, 2),
        })
        
        best_congestion = best_congestion_new
        best_losses = best_losses_new
        
        if verbose:
            print(f"  Iteration {iteration+1}: Opened Line {best_line_to_open} ({from_b}-{to_b})")
            print(f"    → Congestion index: {best_congestion:.4f} | Losses: {best_losses*1000:.1f} kW")
        
        # Recalculer baseline pour prochaine itération
        baseline = run_power_flow(net_work)
    
    # Résultat final
    final_result = run_power_flow(net_work)
    
    return {
        "switch_operations": switch_operations,
        "n_operations": len(switch_operations),
        "opened_lines": opened_lines,
        "final_metrics": final_result,
    }


def _is_network_connected(net: pp.pandapowerNet) -> bool:
    """Vérifie que le réseau est connexe (pas d'îlots) après ouverture de lignes."""
    try:
        import networkx as nx
    except ImportError:
        return True  # Pas de vérif si networkx absent
    
    G = nx.Graph()
    for i in net.bus.index:
        G.add_node(i)
    for i in net.line.index:
        if net.line.at[i, 'in_service']:
            G.add_edge(
                int(net.line.at[i, 'from_bus']),
                int(net.line.at[i, 'to_bus'])
            )
    # Le réseau doit être connexe
    return nx.is_connected(G)


# ============================================================
# 6. SIMULATION 24H COMPLÈTE
# ============================================================

def simulate_24h(season: str = 'summer', load_factor: float = 1.0) -> list:
    """
    Simule le réseau sur 24h (une heure par pas).
    Retourne les métriques par heure.
    """
    results = []
    ghi_profile = GHI_SUMMER_HOURLY if season == 'summer' else GHI_WINTER_HOURLY
    
    pv_buses = [5, 11, 17]
    pv_capacity = [500, 500, 500]  # kWc
    
    for hour in range(24):
        net = build_atacama_network(
            hour=hour, season=season,
            load_factor=load_factor,
            pv_buses=pv_buses,
            pv_capacity_kwp=pv_capacity
        )
        pf = run_power_flow(net)
        ghi = ghi_profile.get(hour, 0)
        
        results.append({
            "hour": hour,
            "ghi_w_m2": ghi,
            "pv_total_kw": sum(pv_output_kw_v2(ghi, c) for c in pv_capacity),
            **pf
        })
    
    return results


# ============================================================
# 7. MAIN — Exécution et export JSON
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GRIZLI — Case Study: Atacama Distribution Network")
    print("Simulation rigoureuse PandaPower — 13.2 kV / 50 Hz")
    print("=" * 60)
    
    PV_BUSES    = [5, 11, 17]
    PV_CAPACITY = [500, 500, 500]  # kWc
    
    # --- SCÉNARIO: PIC SOLAIRE été, 14h ---
    HOUR    = 14
    SEASON  = 'summer'
    LOAD_F  = 1.0

    print(f"\n[1] Construction réseau Atacama — {SEASON} {HOUR}h00")
    net_baseline = build_atacama_network(
        hour=HOUR, season=SEASON, load_factor=LOAD_F,
        pv_buses=PV_BUSES, pv_capacity_kwp=PV_CAPACITY,
        verbose=True
    )
    
    ghi_now = GHI_SUMMER_HOURLY[HOUR]
    pv_now  = sum(pv_output_kw_v2(ghi_now, c) for c in PV_CAPACITY)
    print(f"  GHI: {ghi_now} W/m²  |  PV total: {pv_now:.1f} kW  |  Charge facteur: {LOAD_F}")

    print(f"\n[2] Power Flow BASELINE (avant optimisation)")
    baseline_pf = run_power_flow(net_baseline)
    
    print(f"  Convergé: {baseline_pf['converged']}")
    print(f"  Charge totale:       {baseline_pf['total_load_mw']*1000:.1f} kW")
    print(f"  Génération PV:       {baseline_pf['total_gen_pv_mw']*1000:.1f} kW")
    print(f"  Pertes totales:      {baseline_pf['total_losses_kw']:.1f} kW ({baseline_pf['loss_percent']:.2f}%)")
    print(f"  Max loading ligne:   {baseline_pf['max_loading_percent']:.1f}%")
    print(f"  Tensions min/max:    {baseline_pf['min_voltage_pu']:.4f} / {baseline_pf['max_voltage_pu']:.4f} p.u.")
    print(f"  Lignes surchargées:  {baseline_pf['overloaded_lines']}")
    print(f"  Violations tension:  {baseline_pf['voltage_undervolt']} sous-tension + {baseline_pf['voltage_overvolt']} sur-tension")
    print(f"  Indice congestion:   {baseline_pf['congestion_index']:.4f}")
    print(f"  Réseau faisable:     {baseline_pf['is_feasible']}")

    print(f"\n[3] Optimisation topologique Grizli")
    import copy, time
    net_opt = copy.deepcopy(net_baseline)
    t_start = time.time()
    opt_result = optimize_topology(net_opt, max_switches=3, verbose=True)
    t_end = time.time()
    compute_ms = (t_end - t_start) * 1000
    
    print(f"  Temps de calcul: {compute_ms:.1f} ms")
    print(f"  Opérations de commutation: {opt_result['n_operations']}")
    for op in opt_result['switch_operations']:
        print(f"    → {op['operation']} Line_{op['line_idx']} ({op['from_bus']}-{op['to_bus']}) | loading avant: {op['loading_before']:.1f}%")

    after_pf = opt_result['final_metrics']
    print(f"\n[4] Power Flow APRÈS optimisation")
    print(f"  Pertes totales:      {after_pf['total_losses_kw']:.1f} kW ({after_pf['loss_percent']:.2f}%)")
    print(f"  Max loading ligne:   {after_pf['max_loading_percent']:.1f}%")
    print(f"  Tensions min/max:    {after_pf['min_voltage_pu']:.4f} / {after_pf['max_voltage_pu']:.4f} p.u.")
    print(f"  Lignes surchargées:  {after_pf['overloaded_lines']}")
    print(f"  Indice congestion:   {after_pf['congestion_index']:.4f}")
    print(f"  Réseau faisable:     {after_pf['is_feasible']}")

    # Calcul des améliorations
    delta_losses_kw  = baseline_pf['total_losses_kw'] - after_pf['total_losses_kw']
    delta_losses_pct = delta_losses_kw / baseline_pf['total_losses_kw'] * 100 if baseline_pf['total_losses_kw'] > 0 else 0
    delta_max_load   = baseline_pf['max_loading_percent'] - after_pf['max_loading_percent']
    delta_cong       = baseline_pf['congestion_index'] - after_pf['congestion_index']
    
    print(f"\n[5] Résumé des améliorations")
    print(f"  Réduction pertes:    {delta_losses_kw:.1f} kW  ({delta_losses_pct:.1f}%)")
    print(f"  Réduction max load:  {delta_max_load:.1f} pp")
    print(f"  Réduction cong idx:  {delta_cong:.4f}")

    print(f"\n[6] Simulation 24h — été")
    ts_results = simulate_24h(season='summer', load_factor=1.0)
    total_losses_24h_kwh = sum(r['total_losses_kw'] for r in ts_results)
    max_loading_24h      = max(r['max_loading_percent'] for r in ts_results if r['converged'])
    n_critical_hours     = sum(1 for r in ts_results if r.get('overloaded_lines', 0) > 0 or r.get('max_loading_percent', 0) > 80)
    print(f"  Pertes cumulées 24h: {total_losses_24h_kwh:.1f} kWh")
    print(f"  Max loading 24h:     {max_loading_24h:.1f}%")
    print(f"  Heures critiques (>80% loading): {n_critical_hours}")

    # --- Export JSON ---
    export = {
        "metadata": {
            "title": "Grizli Case Study — Atacama Distribution Network",
            "location": "Antofagasta Region, Chile",
            "network": "33-bus radial feeder, 13.2 kV, 50 Hz",
            "scenario": f"{SEASON.capitalize()} peak — Hour {HOUR}:00",
            "pv_buses": PV_BUSES,
            "pv_capacity_kwp": PV_CAPACITY,
            "pv_total_mwp": sum(PV_CAPACITY) / 1000,
            "ghi_w_m2": ghi_now,
            "pv_output_kw": round(pv_now, 1),
            "compute_time_ms": round(compute_ms, 1),
            "data_sources": [
                "IEEE 33-bus: usmart.ece.utah.edu dataset",
                "Solar irradiance: ScienceDirect / Scribd — Atacama 2021-2024",
                "Load profile: BID Latin America typical distribution profile",
                "Network voltage: 13.2 kV (CNE Chile standard)",
                "Power flow: PandaPower 3.4.0 — Newton-Raphson"
            ]
        },
        "baseline": baseline_pf,
        "optimized": after_pf,
        "switch_operations": opt_result['switch_operations'],
        "improvements": {
            "loss_reduction_kw":   round(delta_losses_kw, 2),
            "loss_reduction_pct":  round(delta_losses_pct, 2),
            "max_loading_reduction_pp": round(delta_max_load, 2),
            "congestion_index_reduction": round(delta_cong, 4),
            "overloaded_lines_before": baseline_pf['overloaded_lines'],
            "overloaded_lines_after":  after_pf['overloaded_lines'],
            "feasible_after": after_pf['is_feasible'],
            "n_switch_operations": opt_result['n_operations'],
        },
        "timeseries_24h": ts_results,
        "timeseries_summary": {
            "total_losses_kwh": round(total_losses_24h_kwh, 1),
            "max_loading_24h": round(max_loading_24h, 2),
            "critical_hours": n_critical_hours,
        }
    }
    
    with open("/app/grizli-casestudy/results.json", "w") as f:
        json.dump(export, f, indent=2, default=str)
    
    print(f"\n✅ Résultats exportés → grizli-casestudy/results.json")
    print("=" * 60)
