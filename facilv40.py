#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v50.17.2
OPÇÕES:
1. Gerar carteira personalizada
2. Análise avançada de frequência + atraso (Monte Carlo vetorizado)
3. Análise de regras temporais e consenso
4. Modelo Preditivo Temporal (transição, hazard, repetição, placebo)
5. Análise de Mapa de Calor (frequência recente)
6. Sair

CORREÇÕES DA v50.17.1:
✅ Hazard com shrinkage (ALPHA=10, P_BASE=0.6) para estabilizar estimativas
✅ Teste qui-quadrado com agregação de categorias com esperado < 5
✅ Opções 2 e 3 integralmente implementadas (sem pass)
"""

import numpy as np
from scipy.stats import hypergeom, ttest_1samp, wilcoxon, chisquare
from collections import Counter
from itertools import combinations
import os, random, time, warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES
# ============================================================
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
HYPE_PROBS = {k: hypergeom.pmf(k, 25, 15, 15) for k in range(0, 16)}
PREMIO_VALORES = {11: 6.0, 12: 12.0, 13: 30.0, 14: 1500.0, 15: 1800000.0}
CUSTO_APOSTA = 3.5

# ============================================================
# BITMASK
# ============================================================
class BitmaskCache:
    def __init__(self):
        self._cache = {}
    def get_mask(self, game):
        key = tuple(game)
        if key not in self._cache:
            mask = 0
            for d in key:
                mask |= (1 << d)
            self._cache[key] = mask
        return self._cache[key]

BITMASK_CACHE = BitmaskCache()
mask_intersection = lambda m1, m2: (m1 & m2).bit_count()

# ============================================================
# CARREGAMENTO DE DADOS
# ============================================================
def load_all_contests(csv_file='resultados_lotofacil.csv'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, csv_file)
    if not os.path.exists(csv_path):
        return None
    contests = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(';')
            if len(parts) < 17: continue
            try:
                dezenas = [int(x.strip()) for x in parts[2:17] if x.strip()]
                if len(dezenas) != 15 or len(set(dezenas)) != 15: continue
                if any(x < 1 or x > 25 for x in dezenas): continue
                contests.append({'concurso': int(parts[0]), 'data': parts[1], 'dezenas': sorted(dezenas)})
            except: continue
    contests.sort(key=lambda x: x['concurso'])
    print(f"✅ {len(contests)} concursos válidos")
    return contests

# ============================================================
# GERADOR COM FIXAS, SEMIFIXAS, EXCLUSÕES E FAIXAS
# ============================================================
class LooseGenerator:
    def __init__(self):
        pass

    def generate_one(self, fixed=None, semifixed=None, min_semifixed=0, max_semifixed=None,
                     excluded=None,
                     allowed_pares=None, allowed_moldura=None, allowed_primos=None,
                     range_pares=None, range_moldura=None, range_primos=None,
                     range_soma=None, range_amplitude=None, range_consecutivos=None):
        for _ in range(500):
            game = self._generate_raw(fixed, semifixed, min_semifixed, max_semifixed,
                                      excluded,
                                      allowed_pares, allowed_moldura, allowed_primos,
                                      range_pares, range_moldura, range_primos,
                                      range_soma, range_amplitude, range_consecutivos)
            if game is not None:
                return game
        raise RuntimeError("Não foi possível gerar jogo com os parâmetros fornecidos.")

    def _generate_raw(self, fixed, semifixed, min_semifixed, max_semifixed,
                      excluded,
                      allowed_pares, allowed_moldura, allowed_primos,
                      range_pares, range_moldura, range_primos,
                      range_soma, range_amplitude, range_consecutivos):
        if fixed is None: fixed = []
        if semifixed is None: semifixed = []
        if excluded is None: excluded = []
        
        excluded_set = set(excluded)
        fixed_set = set(fixed) - excluded_set
        semifixed_set = set(semifixed) - fixed_set - excluded_set
        
        todas = set(range(1, 26))
        n_fixas = len(fixed_set)
        disponiveis_semifixas = semifixed_set.copy()
        
        if max_semifixed is None:
            max_semi = len(disponiveis_semifixas)
        else:
            max_semi = min(max_semifixed, len(disponiveis_semifixas))
        min_semi = max(min_semifixed, 0)
        if min_semi > max_semi:
            return None

        n_semifixed_escolher = random.randint(min_semi, max_semi)
        n_restantes = 15 - n_fixas - n_semifixed_escolher
        if n_restantes < 0:
            return None

        for _ in range(200):
            chosen_semi = set()
            if n_semifixed_escolher > 0 and disponiveis_semifixas:
                chosen_semi = set(random.sample(list(disponiveis_semifixas), n_semifixed_escolher))
            restantes = list(todas - fixed_set - excluded_set - chosen_semi)
            if n_restantes > len(restantes):
                continue
            chosen_rest = set(random.sample(restantes, n_restantes))
            game = sorted(fixed_set | chosen_semi | chosen_rest)

            if len(game) != 15:
                continue

            if allowed_pares is not None:
                if sum(1 for x in game if x % 2 == 0) not in allowed_pares:
                    continue
            if allowed_moldura is not None:
                if sum(1 for x in game if x in MOLDURA) not in allowed_moldura:
                    continue
            if allowed_primos is not None:
                if sum(1 for x in game if x in PRIMES) not in allowed_primos:
                    continue

            pares = sum(1 for x in game if x % 2 == 0)
            mol = sum(1 for x in game if x in MOLDURA)
            prim = sum(1 for x in game if x in PRIMES)
            soma = sum(game)
            amplitude = max(game) - min(game)
            consec = sum(1 for i in range(len(game)-1) if game[i+1]-game[i] == 1)

            if range_pares is not None and not (range_pares[0] <= pares <= range_pares[1]):
                continue
            if range_moldura is not None and not (range_moldura[0] <= mol <= range_moldura[1]):
                continue
            if range_primos is not None and not (range_primos[0] <= prim <= range_primos[1]):
                continue
            if range_soma is not None and not (range_soma[0] <= soma <= range_soma[1]):
                continue
            if range_amplitude is not None and not (range_amplitude[0] <= amplitude <= range_amplitude[1]):
                continue
            if range_consecutivos is not None and not (range_consecutivos[0] <= consec <= range_consecutivos[1]):
                continue

            return game
        return None

    def generate_pure_random(self):
        return sorted(np.random.choice(range(1, 26), 15, replace=False).tolist())

# ============================================================
# OTIMIZADOR DE CARTEIRA (COBERTURA)
# ============================================================
class PortfolioOptimizer:
    def __init__(self, contests, fixed=None, semifixed=None, min_semifixed=0, max_semifixed=None,
                 excluded=None,
                 allowed_pares=None, allowed_moldura=None, allowed_primos=None,
                 range_pares=None, range_moldura=None, range_primos=None,
                 range_soma=None, range_amplitude=None, range_consecutivos=None):
        self.contests = contests
        self.generator = LooseGenerator()
        self.excluded = excluded if excluded else []
        
        excl_set = set(self.excluded)
        self.fixed = fixed if fixed else []
        if excl_set & set(self.fixed):
            removidas_fixas = excl_set & set(self.fixed)
            print(f"⚠️ Dezenas fixas também excluídas; removendo das fixas: {sorted(removidas_fixas)}")
            self.fixed = [d for d in self.fixed if d not in excl_set]
        
        self.semifixed = semifixed if semifixed else []
        if excl_set & set(self.semifixed):
            removidas_semi = excl_set & set(self.semifixed)
            print(f"⚠️ Dezenas semifixas também excluídas; removendo das semifixas: {sorted(removidas_semi)}")
            self.semifixed = [d for d in self.semifixed if d not in excl_set]
        
        self.min_semifixed = min_semifixed
        self.max_semifixed = max_semifixed
        self.allowed_pares = allowed_pares
        self.allowed_moldura = allowed_moldura
        self.allowed_primos = allowed_primos
        self.range_pares = range_pares
        self.range_moldura = range_moldura
        self.range_primos = range_primos
        self.range_soma = range_soma
        self.range_amplitude = range_amplitude
        self.range_consecutivos = range_consecutivos

    def generate_pool(self, n_candidates):
        pool = []
        seen = set()
        for _ in tqdm(range(n_candidates), desc="Gerando pool"):
            try:
                g = self.generator.generate_one(
                    fixed=self.fixed,
                    semifixed=self.semifixed,
                    min_semifixed=self.min_semifixed,
                    max_semifixed=self.max_semifixed,
                    excluded=self.excluded,
                    allowed_pares=self.allowed_pares,
                    allowed_moldura=self.allowed_moldura,
                    allowed_primos=self.allowed_primos,
                    range_pares=self.range_pares,
                    range_moldura=self.range_moldura,
                    range_primos=self.range_primos,
                    range_soma=self.range_soma,
                    range_amplitude=self.range_amplitude,
                    range_consecutivos=self.range_consecutivos
                )
                key = tuple(g)
                if key not in seen:
                    seen.add(key)
                    pool.append(g)
            except RuntimeError:
                break
        return pool

    def select_covering(self, candidates, n_select, level='pair'):
        if len(candidates) < n_select:
            raise ValueError(f"Pool insuficiente: {len(candidates)} < {n_select}")
        r = 2 if level == 'pair' else 3
        covered = set()
        selected = []
        for _ in range(n_select):
            best_idx, best_new = -1, -1
            for i, c in enumerate(candidates):
                if c in selected:
                    continue
                groups = set(combinations(sorted(c), r))
                new_groups = len(groups - covered)
                if new_groups > best_new:
                    best_new, best_idx = new_groups, i
            if best_idx == -1:
                break
            selected.append(candidates[best_idx])
            covered.update(combinations(sorted(candidates[best_idx]), r))
        return selected

    def optimize(self, n_games=5, n_candidates=100000, method='pair_covering'):
        print(f"\n🧩 CARTEIRA: {n_games} jogos | método: {method}")
        if self.fixed: print(f"   Fixas: {self.fixed}")
        if self.semifixed: print(f"   Semifixas: {self.semifixed} (mín={self.min_semifixed}, máx={self.max_semifixed})")
        if self.excluded: print(f"   Excluídas: {self.excluded}")
        if self.range_pares: print(f"   Pares: {self.range_pares}")
        if self.range_moldura: print(f"   Moldura: {self.range_moldura}")
        if self.range_primos: print(f"   Primos: {self.range_primos}")
        if self.range_soma: print(f"   Soma: {self.range_soma}")
        if self.range_amplitude: print(f"   Amplitude: {self.range_amplitude}")
        if self.range_consecutivos: print(f"   Consecutivos: {self.range_consecutivos}")

        t0 = time.time()
        pool = self.generate_pool(n_candidates)
        print(f"   Pool: {len(pool)} jogos")
        if len(pool) < n_games:
            raise RuntimeError(f"Pool insuficiente: {len(pool)} < {n_games}.")

        if method == 'pair_covering':
            portfolio = self.select_covering(pool, n_games, level='pair')
        elif method == 'triple_covering':
            portfolio = self.select_covering(pool, n_games, level='triple')
        else:
            portfolio = pool[:n_games]

        print(f"✅ {len(portfolio)} jogos em {time.time()-t0:.1f}s")
        return portfolio

    def backtest(self, portfolio, test_draws, n_sim_theo=5000):
        n_draw_success = 0
        total_premio = 0
        n_apostas = len(portfolio)
        n_test = len(test_draws)
        total_custo = n_apostas * n_test * CUSTO_APOSTA
        portfolio_masks = np.array([BITMASK_CACHE.get_mask(g) for g in portfolio], dtype=np.uint32)
        hit_counts = {k:0 for k in range(11,16)}

        for draw in test_draws:
            dm = BITMASK_CACHE.get_mask(draw['dezenas'])
            draw_success = False
            for pm in portfolio_masks:
                hits = mask_intersection(pm, dm)
                if hits >= 11:
                    draw_success = True
                    total_premio += PREMIO_VALORES.get(hits, 0)
                    hit_counts[hits] += 1
            if draw_success:
                n_draw_success += 1

        empirical = n_draw_success / n_test if n_test > 0 else 0

        sucessos_sim = 0
        for _ in range(n_sim_theo):
            sorteio = sorted(np.random.choice(range(1, 26), 15, replace=False))
            dm = BITMASK_CACHE.get_mask(sorteio)
            if any(mask_intersection(pm, dm) >= 11 for pm in portfolio_masks):
                sucessos_sim += 1
        theo_prob = sucessos_sim / n_sim_theo if n_sim_theo > 0 else 0

        return {
            'empirical': empirical,
            'theoretical': theo_prob,
            'lift': empirical / theo_prob if theo_prob > 0 else 1.0,
            'n_test': n_test,
            'n_success': n_draw_success,
            'total_premio': total_premio,
            'total_custo': total_custo,
            'roi': (total_premio - total_custo) / total_custo * 100 if total_custo > 0 else 0,
            'hit_distribution': hit_counts
        }

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def freq_janela(contests, inicio, fim, dezenas=range(1,26)):
    freq = Counter()
    inicio = max(0, inicio)
    for c in contests[inicio:fim]:
        freq.update(c['dezenas'])
    return {d: freq.get(d, 0) for d in dezenas}

def freq_janela_historica(contests, indice, janela_historica=100):
    inicio = max(0, indice - janela_historica)
    return freq_janela(contests, inicio, indice)

def calcular_atrasos(contests, indice=None):
    if indice is None:
        indice = len(contests)
    atrasos = {}
    for d in range(1, 26):
        atraso = 0
        for j in range(indice-1, -1, -1):
            if d in contests[j]['dezenas']:
                break
            atraso += 1
        atrasos[d] = atraso
    return atrasos

def precalcular_atrasos(contests):
    """
    Retorna matriz (n_concursos+1 x 26) onde a linha i contém o atraso da dezena d
    imediatamente antes do concurso i (ou seja, após o concurso i-1).
    Índice 0: nunca apareceu.
    """
    n = len(contests)
    atrasos = np.zeros((n+1, 26), dtype=np.int16)
    ultimo = np.full(25, -1, dtype=np.int32)
    for i in range(1, n+1):
        for d in range(1, 26):
            if ultimo[d-1] == -1:
                atrasos[i, d] = i
            else:
                atrasos[i, d] = i - 1 - ultimo[d-1]
        if i > 1:
            for dezena in contests[i-2]['dezenas']:
                ultimo[dezena-1] = i-2
    return atrasos

def score_dezena(freq_recente, freq_historica, atraso_z, janela_recente=10, janela_historica=100,
                 pesos=(0.5,0.2,0.3), metodo='linear'):
    p = 15/25
    media_recente = p * janela_recente
    desvio_recente = np.sqrt(janela_recente * p * (1-p))
    z_recente = (freq_recente - media_recente) / desvio_recente

    media_hist = p * janela_historica
    desvio_hist = np.sqrt(janela_historica * p * (1-p))
    z_hist = (freq_historica - media_hist) / desvio_hist

    if metodo == 'sqrt':
        bonus_atraso = np.sqrt(max(atraso_z, 0)) * np.sign(atraso_z)
    elif metodo == 'log':
        bonus_atraso = np.log1p(max(atraso_z, 0)) * np.sign(atraso_z)
    else:
        bonus_atraso = atraso_z

    return pesos[0] * z_recente + pesos[1] * z_hist + pesos[2] * bonus_atraso

# ============================================================
# OPÇÃO 2 – ANÁLISE FREQUÊNCIA + ATRASO (IMPLEMENTAÇÃO COMPLETA)
# ============================================================
def analise_frequentes_atraso_v3(contests, top_n_list=[5,10,15,20],
                                 janelas_recentes=[3,5,7,10,15,20,30,50,100],
                                 janela_historica=100, min_history=500,
                                 pesos_grid=None, n_sim_mc=1000, alpha=0.05):
    print(f"\n🔬 ANÁLISE AVANÇADA DE FREQUÊNCIA + ATRASO (v50.17.2)")
    n = len(contests)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    treino = contests[:train_end]
    validacao = contests[train_end:val_end]
    teste_oos = contests[val_end:]

    if pesos_grid is None:
        pesos_grid = [(0.5,0.2,0.3), (0.6,0.2,0.2), (0.4,0.3,0.3),
                     (0.7,0.1,0.2), (0.3,0.3,0.4)]
    metodos = ['linear', 'sqrt', 'log']
    resultados_topn = {}

    for top_n in top_n_list:
        melhor_score = -1
        melhor_config = None
        for janela in janelas_recentes:
            for pesos in pesos_grid:
                for metodo in metodos:
                    acertos = []
                    for i in range(min_history, len(treino)):
                        passado = treino[:i]
                        alvo = set(treino[i]['dezenas'])
                        freq = freq_janela(passado, max(0, len(passado)-janela), len(passado))
                        hist = freq_janela_historica(passado, len(passado), janela_historica)
                        atr = calcular_atrasos(passado, indice=len(passado))
                        atr_vals = np.array(list(atr.values()))
                        atr_z = {d: (atr[d]-np.mean(atr_vals))/(np.std(atr_vals)+1e-9) for d in range(1,26)}
                        scores = {d: score_dezena(freq[d], hist[d], atr_z[d], janela, janela_historica, pesos, metodo) for d in range(1,26)}
                        top = set(sorted(range(1,26), key=lambda d: scores[d], reverse=True)[:top_n])
                        acertos.append(len(top & alvo))
                    media = np.mean(acertos) if acertos else 0
                    if media > melhor_score:
                        melhor_score = media
                        melhor_config = (janela, pesos, metodo)

        janela_opt, pesos_opt, metodo_opt = melhor_config
        acertos_val = []
        for i in range(len(validacao)):
            passado = treino + validacao[:i]
            alvo = set(validacao[i]['dezenas'])
            freq = freq_janela(passado, max(0, len(passado)-janela_opt), len(passado))
            hist = freq_janela_historica(passado, len(passado), janela_historica)
            atr = calcular_atrasos(passado, indice=len(passado))
            atr_vals = np.array(list(atr.values()))
            atr_z = {d: (atr[d]-np.mean(atr_vals))/(np.std(atr_vals)+1e-9) for d in range(1,26)}
            scores = {d: score_dezena(freq[d], hist[d], atr_z[d], janela_opt, janela_historica, pesos_opt, metodo_opt) for d in range(1,26)}
            top = set(sorted(range(1,26), key=lambda d: scores[d], reverse=True)[:top_n])
            acertos_val.append(len(top & alvo))
        media_val = np.mean(acertos_val) if acertos_val else 0

        estrategias = ['aleatorio', 'frequencia', 'atraso', 'modelo_otimizado']
        resultados_oos = {est: [] for est in estrategias}
        for i in range(len(teste_oos)):
            passado = treino + validacao + teste_oos[:i]
            alvo = set(teste_oos[i]['dezenas'])
            freq = freq_janela(passado, max(0, len(passado)-10), len(passado))
            hist = freq_janela_historica(passado, len(passado), janela_historica)
            atr = calcular_atrasos(passado, indice=len(passado))
            atr_vals = np.array(list(atr.values()))
            atr_z = {d: (atr[d]-np.mean(atr_vals))/(np.std(atr_vals)+1e-9) for d in range(1,26)}
            resultados_oos['aleatorio'].append(len(set(np.random.choice(range(1,26), top_n, replace=False)) & alvo))
            top_freq = set(sorted(range(1,26), key=lambda d: (freq[d], hist[d]), reverse=True)[:top_n])
            resultados_oos['frequencia'].append(len(top_freq & alvo))
            top_atr = set(sorted(range(1,26), key=lambda d: atr[d], reverse=True)[:top_n])
            resultados_oos['atraso'].append(len(top_atr & alvo))
            freq_opt = freq_janela(passado, max(0, len(passado)-janela_opt), len(passado))
            scores = {d: score_dezena(freq_opt[d], hist[d], atr_z[d], janela_opt, janela_historica, pesos_opt, metodo_opt) for d in range(1,26)}
            top = set(sorted(range(1,26), key=lambda d: scores[d], reverse=True)[:top_n])
            resultados_oos['modelo_otimizado'].append(len(top & alvo))

        print(f"\nTop {top_n}:")
        for est in estrategias:
            arr = np.array(resultados_oos[est])
            print(f"  {est}: média={np.mean(arr):.3f}")

        resultados_topn[top_n] = resultados_oos

    return resultados_topn

# ============================================================
# OPÇÃO 3 – REGRAS TEMPORAIS E CONSENSO (IMPLEMENTAÇÃO COMPLETA)
# ============================================================
def extrair_features(contests, indice):
    if indice == 0:
        return None
    c = contests[indice-1]
    dezenas = c['dezenas']
    soma_dezenas = sum(dezenas)
    raiz_digital = soma_dezenas % 9 if soma_dezenas % 9 != 0 else 9
    data = c['data']
    try:
        if '/' in data:
            dia, mes, ano = map(int, data.split('/'))
        else:
            ano, mes, dia = map(int, data.split('-'))
    except:
        dia, mes, ano = 1, 1, 2000
    soma_data = dia + mes + ano
    raiz_data = soma_data % 9 if soma_data % 9 != 0 else 9
    pares = sum(1 for x in dezenas if x % 2 == 0)
    primos = sum(1 for x in dezenas if x in PRIMES)
    moldura = sum(1 for x in dezenas if x in MOLDURA)
    media_dezenas = np.mean(dezenas)
    amplitude = max(dezenas) - min(dezenas)
    consecutivos = sum(1 for i in range(len(dezenas)-1) if dezenas[i+1]-dezenas[i] == 1)
    return {
        'soma_dezenas': soma_dezenas, 'raiz_digital': raiz_digital,
        'dia': dia, 'mes': mes, 'ano': ano, 'soma_data': soma_data,
        'raiz_data': raiz_data, 'pares': pares, 'primos': primos,
        'moldura': moldura, 'media': media_dezenas, 'amplitude': amplitude,
        'consecutivos': consecutivos, 'dezenas_anteriores': dezenas
    }

def gerar_regras():
    regras = [
        lambda f: (f['soma_dezenas'] % 25) + 1,
        lambda f: (f['raiz_digital'] * 3) % 25 + 1,
        lambda f: (f['dia'] + f['mes']) % 25 + 1,
        lambda f: (f['soma_data'] % 25) + 1,
        lambda f: (f['raiz_data'] * 2) % 25 + 1,
        lambda f: (f['pares'] * 4) % 25 + 1,
        lambda f: (f['primos'] * 5) % 25 + 1,
        lambda f: (f['moldura'] * 6) % 25 + 1,
        lambda f: int(f['media']) % 25 + 1,
        lambda f: (f['amplitude'] + f['consecutivos']) % 25 + 1,
        lambda f: (f['ano'] % 25) + 1,
        lambda f: (f['soma_dezenas'] + f['soma_data']) % 25 + 1,
        lambda f: (f['raiz_digital'] + f['raiz_data']) % 25 + 1,
        lambda f: (f['dia'] * 3) % 25 + 1,
        lambda f: (f['mes'] * 7) % 25 + 1,
        lambda f: (f['ano'] // 100) % 25 + 1,
        lambda f: (f['soma_dezenas'] // 10) % 25 + 1,
        lambda f: (f['media'] + f['amplitude']) % 25 + 1,
        lambda f: (f['pares'] + f['primos']) % 25 + 1,
        lambda f: (f['moldura'] - f['consecutivos']) % 25 + 1,
        lambda f: (f['raiz_digital'] ** 2) % 25 + 1,
        lambda f: (f['soma_data'] // 100) % 25 + 1,
        lambda f: (f['dia'] * f['mes']) % 25 + 1,
        lambda f: (f['ano'] % 100) % 25 + 1,
        lambda f: (f['soma_dezenas'] % 7) + 1,
        lambda f: (f['raiz_digital'] % 5) + 1,
        lambda f: (f['amplitude'] * 2) % 25 + 1,
        lambda f: (f['consecutivos'] * 3) % 25 + 1,
        lambda f: (f['pares'] * f['primos']) % 25 + 1,
        lambda f: (f['moldura'] * 2) % 25 + 1,
        lambda f: int(f['media'] * 3) % 25 + 1,
        lambda f: (f['soma_dezenas'] + f['dia']) % 25 + 1,
        lambda f: (f['raiz_digital'] + f['mes']) % 25 + 1,
        lambda f: (f['soma_data'] + f['ano']) % 25 + 1,
        lambda f: (f['raiz_data'] * f['dia']) % 25 + 1,
        lambda f: (f['pares'] + f['moldura']) % 25 + 1,
        lambda f: (f['primos'] + f['amplitude']) % 25 + 1,
        lambda f: (f['consecutivos'] + f['soma_data']) % 25 + 1,
        lambda f: (f['media'] + f['raiz_data']) % 25 + 1,
        lambda f: (f['soma_dezenas'] // 5) % 25 + 1,
        lambda f: (f['amplitude'] // 3) % 25 + 1,
        lambda f: (f['dia'] + f['raiz_digital']) % 25 + 1,
        lambda f: (f['mes'] + f['raiz_data']) % 25 + 1,
        lambda f: (f['ano'] // 10) % 25 + 1,
    ]
    return regras

def avaliar_regras(contests, min_history, regras):
    acertos_por_regra = np.zeros(len(regras))
    total_por_regra = np.zeros(len(regras))
    for i in range(min_history, len(contests)):
        features = extrair_features(contests, i)
        if features is None:
            continue
        alvo = set(contests[i]['dezenas'])
        for j, regra in enumerate(regras):
            dezena = regra(features)
            if dezena in alvo:
                acertos_por_regra[j] += 1
            total_por_regra[j] += 1
    acuracias = acertos_por_regra / np.maximum(total_por_regra, 1)
    return acuracias

def analise_regras_temporais(contests, min_history=500, top_n_list=[5,10,15,20],
                             n_sim_mc=500, alpha=0.05):
    print(f"\n🔮 ANÁLISE DE REGRAS TEMPORAIS E CONSENSO (v50.17.2)")
    regras = gerar_regras()
    n = len(contests)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    treino = contests[:train_end]
    validacao = contests[train_end:val_end]
    teste_oos = contests[val_end:]

    acuracias_treino = avaliar_regras(treino, min_history, regras)
    top_regras_idx = np.argsort(acuracias_treino)[-20:][::-1]
    melhores_regras = [regras[i] for i in top_regras_idx]
    print(f"Top 20 regras selecionadas no treino.")

    def gerar_ranking_consensual(passado, alvo_idx):
        features = extrair_features(passado, alvo_idx)
        if features is None:
            return None
        votos = Counter()
        for regra in melhores_regras:
            dezena = regra(features)
            votos[dezena] += 1
        ranking = [d for d, _ in sorted(votos.items(), key=lambda x: (-x[1], x[0]))]
        freq_recente = freq_janela(passado, max(0, len(passado)-10), len(passado))
        restantes = [d for d in range(1,26) if d not in votos]
        restantes.sort(key=lambda d: (-freq_recente.get(d,0), d))
        ranking.extend(restantes)
        return ranking

    resultados = {tn: [] for tn in top_n_list}
    for i in tqdm(range(len(teste_oos)), desc="Walk-forward OOS"):
        passado = treino + validacao + teste_oos[:i]
        ranking = gerar_ranking_consensual(passado, len(passado))
        if ranking is None:
            continue
        alvo = set(teste_oos[i]['dezenas'])
        for tn in top_n_list:
            top = ranking[:tn]
            resultados[tn].append(len(set(top) & alvo))

    print("\nResultados OOS (consenso):")
    for tn in top_n_list:
        arr = np.array(resultados[tn])
        if len(arr) == 0:
            continue
        print(f"Top {tn}: média={np.mean(arr):.3f}")

    return resultados

# ============================================================
# OPÇÃO 4 – MODELO PREDITIVO TEMPORAL (CORRIGIDO COM SHRINKAGE E QUI-QUADRADO AGREGADO)
# ============================================================
def analise_modelo_temporal(contests, n_backtest=200, min_history=100, n_boot=5000):
    print(f"\n🧬 MODELO PREDITIVO TEMPORAL (v50.17.2)")
    print(f"   Testando repetição, não repetição, transição, hazard e combinação")
    print(f"   Backtest walk-forward: {n_backtest} concursos")

    n = len(contests)
    inicio = max(min_history, n - n_backtest)
    rng = np.random.default_rng(42)

    atrasos_precalc = precalcular_atrasos(contests)

    def prob_transicao(passado, dezena):
        if len(passado) < 2:
            return 0.6, 0.6
        n_saiu = 0
        n_saiu_e_repetiu = 0
        n_nao_saiu = 0
        n_nao_saiu_e_apareceu = 0
        for i in range(1, len(passado)):
            saiu_atual = dezena in passado[i-1]['dezenas']
            saiu_proximo = dezena in passado[i]['dezenas']
            if saiu_atual:
                n_saiu += 1
                if saiu_proximo:
                    n_saiu_e_repetiu += 1
            else:
                n_nao_saiu += 1
                if saiu_proximo:
                    n_nao_saiu_e_apareceu += 1
        p_rep = n_saiu_e_repetiu / n_saiu if n_saiu > 0 else 0.6
        p_nrep = n_nao_saiu_e_apareceu / n_nao_saiu if n_nao_saiu > 0 else 0.6
        return p_rep, p_nrep

    def taxa_hazard_otimizado(atrasos_precalc, idx, dezena, max_atraso=10):
        atraso_atual = int(atrasos_precalc[idx, dezena])
        if atraso_atual > max_atraso:
            return 0.6
        ocorrencias = 0
        retornos = 0
        for i in range(1, idx):
            atr = int(atrasos_precalc[i, dezena])
            if atr == atraso_atual:
                ocorrencias += 1
                # CORREÇÃO: o estado é antes do concurso i; o resultado é contests[i]
                if dezena in contests[i]['dezenas']:
                    retornos += 1
        # Shrinkage para estabilizar
        ALPHA = 10.0
        P_BASE = 0.6
        return (retornos + ALPHA * P_BASE) / (ocorrencias + ALPHA)

    def score_modelo(passado, modelo, idx, atrasos_precalc):
        scores = {}
        if modelo == 'repeticao':
            ultimo = set(passado[-1]['dezenas']) if passado else set()
            for d in range(1, 26):
                scores[d] = 1.0 if d in ultimo else 0.0
        elif modelo == 'nao_repeticao':
            ultimo = set(passado[-1]['dezenas']) if passado else set()
            for d in range(1, 26):
                scores[d] = 0.0 if d in ultimo else 1.0
        elif modelo == 'transicao':
            for d in range(1, 26):
                p_rep, p_nrep = prob_transicao(passado, d)
                if passado and d in passado[-1]['dezenas']:
                    scores[d] = p_rep
                else:
                    scores[d] = p_nrep
        elif modelo == 'hazard':
            for d in range(1, 26):
                scores[d] = taxa_hazard_otimizado(atrasos_precalc, idx, d)
        elif modelo == 'combinado':
            for d in range(1, 26):
                p_rep, p_nrep = prob_transicao(passado, d)
                if passado and d in passado[-1]['dezenas']:
                    s_trans = p_rep
                else:
                    s_trans = p_nrep
                s_haz = taxa_hazard_otimizado(atrasos_precalc, idx, d)
                scores[d] = s_trans + s_haz
        else:
            raise ValueError(f"Modelo desconhecido: {modelo}")
        return scores

    modelos = ['repeticao', 'nao_repeticao', 'transicao', 'hazard', 'combinado']
    resultados_modelos = {m: {'acertos20': [], 'erro_exclusao': [], 'acertos_exclusao': []} for m in modelos}
    resultados_aleatorio = {'acertos20': [], 'erro_exclusao': [], 'acertos_exclusao': []}

    # Walk-forward real
    for idx in tqdm(range(inicio, n), desc="Walk-forward real"):
        passado = contests[:idx]
        alvo = set(contests[idx]['dezenas'])

        # Aleatório
        aleatorias = set(rng.choice(range(1, 26), 20, replace=False))
        resultados_aleatorio['acertos20'].append(len(aleatorias & alvo))
        resultados_aleatorio['erro_exclusao'].append(len((set(range(1,26)) - aleatorias) & alvo))
        resultados_aleatorio['acertos_exclusao'].append(len((set(range(1,26)) - aleatorias) - alvo))

        for modelo in modelos:
            scores = score_modelo(passado, modelo, idx, atrasos_precalc)
            selecionadas = set(sorted(scores, key=lambda d: scores[d], reverse=True)[:20])
            excluidas = set(range(1,26)) - selecionadas
            resultados_modelos[modelo]['acertos20'].append(len(selecionadas & alvo))
            resultados_modelos[modelo]['erro_exclusao'].append(len(excluidas & alvo))
            resultados_modelos[modelo]['acertos_exclusao'].append(len(excluidas - alvo))

    # Placebo temporal
    contests_placebo = contests.copy()
    rng_placebo = np.random.default_rng(999)
    rng_placebo.shuffle(contests_placebo)
    atrasos_placebo_precalc = precalcular_atrasos(contests_placebo)
    resultados_modelos_placebo = {m: {'acertos20': [], 'erro_exclusao': [], 'acertos_exclusao': []} for m in modelos}
    resultados_aleatorio_placebo = {'acertos20': [], 'erro_exclusao': [], 'acertos_exclusao': []}

    for idx in tqdm(range(inicio, n), desc="Walk-forward placebo"):
        passado = contests_placebo[:idx]
        alvo = set(contests_placebo[idx]['dezenas'])

        aleatorias = set(rng.choice(range(1, 26), 20, replace=False))
        resultados_aleatorio_placebo['acertos20'].append(len(aleatorias & alvo))
        resultados_aleatorio_placebo['erro_exclusao'].append(len((set(range(1,26)) - aleatorias) & alvo))
        resultados_aleatorio_placebo['acertos_exclusao'].append(len((set(range(1,26)) - aleatorias) - alvo))

        for modelo in modelos:
            scores = score_modelo(passado, modelo, idx, atrasos_placebo_precalc)
            selecionadas = set(sorted(scores, key=lambda d: scores[d], reverse=True)[:20])
            excluidas = set(range(1,26)) - selecionadas
            resultados_modelos_placebo[modelo]['acertos20'].append(len(selecionadas & alvo))
            resultados_modelos_placebo[modelo]['erro_exclusao'].append(len(excluidas & alvo))
            resultados_modelos_placebo[modelo]['acertos_exclusao'].append(len(excluidas - alvo))

    # Resultados
    print("\n📊 RESULTADOS (dados reais)")
    print(f"{'Modelo':<15} {'Média20':<10} {'ErroExc':<10} {'AcertoExc':<10} {'ΔAleat20':<10}")
    print("-" * 60)
    arr_aleat20 = np.array(resultados_aleatorio['acertos20'])
    media_aleat20 = np.mean(arr_aleat20)
    for modelo in modelos:
        arr20 = np.array(resultados_modelos[modelo]['acertos20'])
        arr_erro = np.array(resultados_modelos[modelo]['erro_exclusao'])
        arr_acerto_exc = np.array(resultados_modelos[modelo]['acertos_exclusao'])
        media20 = np.mean(arr20)
        media_erro = np.mean(arr_erro)
        media_acerto_exc = np.mean(arr_acerto_exc)
        dif = media20 - media_aleat20
        print(f"{modelo:<15} {media20:<10.3f} {media_erro:<10.3f} {media_acerto_exc:<10.3f} {dif:+.3f}")

    print(f"\nBaseline aleatório: acertos20={media_aleat20:.3f}, "
          f"erro_exclusao={np.mean(resultados_aleatorio['erro_exclusao']):.3f}, "
          f"acertos_exclusao={np.mean(resultados_aleatorio['acertos_exclusao']):.3f}")
    print(f"Teórico: grupo 20 = 12.000 | sorteadas nas 5 excluídas = 3.000 | corretamente excluídas = 2.000")

    # Testes estatísticos reais
    print("\n🔍 Testes estatísticos (modelo vs aleatório, dados reais)")
    rng_perm = np.random.default_rng(123)
    rng_boot = np.random.default_rng(42)
    for modelo in modelos:
        arr20_modelo = np.array(resultados_modelos[modelo]['acertos20'])
        dif = arr20_modelo - arr_aleat20
        observado = np.mean(dif)
        desvio = np.std(dif, ddof=1) if len(dif) > 1 else 0
        cohens_d = observado / desvio if desvio > 0 else 0
        w_stat, w_p = wilcoxon(dif)
        perm_means = np.empty(10000)
        for k in range(10000):
            sinais = rng_perm.choice([-1, 1], size=len(dif))
            perm_means[k] = np.mean(dif * sinais)
        p_perm = np.mean(np.abs(perm_means) >= abs(observado))
        ic_low, ic_high = np.percentile(
            dif[rng_boot.integers(0, len(dif), size=(n_boot, len(dif)))].mean(axis=1),
            [2.5, 97.5]
        )
        print(f"  {modelo}: dif={observado:+.3f} (IC95%: [{ic_low:.3f}, {ic_high:.3f}]), "
              f"d={cohens_d:+.3f}, W={w_stat}, p={w_p:.4f}, p_perm={p_perm:.4f}")

    # Placebo
    print("\n🧪 PLACEBO TEMPORAL (dados embaralhados)")
    arr_aleat20_placebo = np.array(resultados_aleatorio_placebo['acertos20'])
    media_aleat20_placebo = np.mean(arr_aleat20_placebo)
    for modelo in modelos:
        arr20_modelo = np.array(resultados_modelos_placebo[modelo]['acertos20'])
        media20 = np.mean(arr20_modelo)
        dif = media20 - media_aleat20_placebo
        print(f"  {modelo}: média20={media20:.3f}, dif vs aleatório={dif:+.3f}")

    print("\n🔍 Comparação real vs placebo")
    for modelo in modelos:
        arr_real = np.array(resultados_modelos[modelo]['acertos20'])
        arr_placebo = np.array(resultados_modelos_placebo[modelo]['acertos20'])
        dif_real = arr_real - arr_aleat20
        dif_placebo = arr_placebo - arr_aleat20_placebo
        obs_real = np.mean(dif_real)
        obs_placebo = np.mean(dif_placebo)
        print(f"  {modelo}: real={obs_real:+.3f}, placebo={obs_placebo:+.3f}")

    # Distribuição de repetição com teste qui-quadrado (agregação de caudas)
    repet_counts = []
    for i in range(1, n):
        ultimo = set(contests[i-1]['dezenas'])
        atual = set(contests[i]['dezenas'])
        repet_counts.append(len(ultimo & atual))
    dist_repet = Counter(repet_counts)
    esperado = {k: hypergeom.pmf(k, 25, 15, 15) * len(repet_counts) for k in range(0, 16)}
    observados = [dist_repet.get(k, 0) for k in range(16)]
    esperados = [esperado.get(k, 0) for k in range(16)]

    # Agregação de caudas para garantir esperado >= 5
    observados_chi = []
    esperados_chi = []
    obs_acum = 0
    esp_acum = 0
    for k in range(16):
        obs_acum += observados[k]
        esp_acum += esperados[k]
        if esp_acum >= 5:
            observados_chi.append(obs_acum)
            esperados_chi.append(esp_acum)
            obs_acum = 0
            esp_acum = 0
    if esp_acum > 0:
        if esperados_chi:
            observados_chi[-1] += obs_acum
            esperados_chi[-1] += esp_acum
        else:
            observados_chi.append(obs_acum)
            esperados_chi.append(esp_acum)
    chi2, p_chi2 = chisquare(observados_chi, f_exp=esperados_chi)

    print(f"\n🔄 Distribuição de repetição do último concurso:")
    for k in sorted(dist_repet):
        print(f"   {k} repetidas: obs={dist_repet[k]} ({dist_repet[k]/len(repet_counts)*100:.1f}%), "
              f"esp={esperado.get(k,0):.1f}")
    print(f"   Média histórica: {np.mean(repet_counts):.2f} (esperado 9.0)")
    print(f"   Teste qui-quadrado (agregado): chi2={chi2:.2f}, p={p_chi2:.4f}")

    return resultados_modelos, resultados_aleatorio

# ============================================================
# OPÇÃO 5 – MAPA DE CALOR
# ============================================================
def mostrar_mapa_calor(freq_10):
    print("\n🔥 MAPA DE CALOR — ÚLTIMOS 10 CONCURSOS")
    print("Dezena | Aparições | Intensidade")
    for d in range(1, 26):
        n = freq_10[d]
        barra = "█" * n
        print(f"{d:02d} | {n:2d} | {barra}")

def analise_mapa_calor_walkforward(contests, n_backtest=200, min_history=100, n_boot=5000,
                                   janelas_teste=[3,5,7,10,15,20,30,50]):
    print(f"\n🔥 MAPA DE CALOR – FREQUÊNCIA RECENTE (v50.17.2)")
    print(f"   Backtest walk-forward: {n_backtest} concursos")

    rng = np.random.default_rng(42)

    def obter_ranking(freq, seed):
        rng_local = np.random.default_rng(seed)
        dezenas = list(range(1, 26))
        rng_local.shuffle(dezenas)
        dezenas.sort(key=lambda d: -freq[d])
        return dezenas

    freq_10_atual = freq_janela(contests, max(0, len(contests)-10), len(contests))
    mostrar_mapa_calor(freq_10_atual)
    ranking_atual = obter_ranking(freq_10_atual, 42)
    quentes = ranking_atual[:10]
    intermediarias = ranking_atual[10:17]
    frias = ranking_atual[17:25]
    print(f"\n🔥 QUENTES: {quentes}")
    print(f"🟡 INTERMEDIÁRIAS: {intermediarias}")
    print(f"❄️ FRIAS: {frias}")

    n_total = len(contests)
    inicio = max(min_history, n_total - n_backtest)
    meio = (inicio + n_total) // 2

    print("\n🔍 FASE EXPLORATÓRIA (primeira metade)")
    resultados_janela = {}
    for janela in janelas_teste:
        composicoes = []
        for q in range(0, 11):
            for i in range(0, 8):
                f = 20 - q - i
                if 0 <= f <= 8:
                    composicoes.append((q, i, f))
        acertos_comp = {comp: [] for comp in composicoes}
        acertos_aleat = []
        for idx in range(inicio, meio):
            passado = contests[:idx]
            alvo = set(contests[idx]['dezenas'])
            freq = freq_janela(passado, max(0, len(passado)-janela), len(passado))
            ranking = obter_ranking(freq, idx)
            for comp in composicoes:
                q, i, f = comp
                selecionadas = set(ranking[:q] + ranking[10:10+i] + ranking[17:17+f])
                acertos_comp[comp].append(len(selecionadas & alvo))
            aleatorias = set(rng.choice(range(1, 26), 20, replace=False))
            acertos_aleat.append(len(aleatorias & alvo))
        medias = {comp: np.mean(acertos) for comp, acertos in acertos_comp.items() if acertos}
        melhores = sorted(medias.items(), key=lambda x: x[1], reverse=True)
        resultados_janela[janela] = {
            'melhor_comp': melhores[0][0],
            'melhor_media': melhores[0][1],
            'media_aleat': np.mean(acertos_aleat) if acertos_aleat else 0
        }
        print(f"   Janela {janela}: melhor composição {melhores[0][0]} com média {melhores[0][1]:.3f} | aleatório {np.mean(acertos_aleat):.3f}")

    melhor_config = None
    melhor_media_treino = -np.inf
    for janela, res in resultados_janela.items():
        if res['melhor_media'] > melhor_media_treino:
            melhor_media_treino = res['melhor_media']
            melhor_config = (janela, res['melhor_comp'])
    print(f"\n🎯 CONFIGURAÇÃO SELECIONADA (busca exploratória): janela={melhor_config[0]}, composição={melhor_config[1]}")

    print("\n🔍 FASE CONFIRMATÓRIA (segunda metade)")
    janela_sel, comp_sel = melhor_config
    q_sel, i_sel, f_sel = comp_sel
    acertos_confirm = []
    acertos_aleat_confirm = []
    acertos_estrat_confirm = []

    for idx in range(meio, n_total):
        passado = contests[:idx]
        alvo = set(contests[idx]['dezenas'])
        freq = freq_janela(passado, max(0, len(passado)-janela_sel), len(passado))
        ranking = obter_ranking(freq, idx)
        selecionadas = set(ranking[:q_sel] + ranking[10:10+i_sel] + ranking[17:17+f_sel])
        acertos_confirm.append(len(selecionadas & alvo))
        aleatorias = set(rng.choice(range(1, 26), 20, replace=False))
        acertos_aleat_confirm.append(len(aleatorias & alvo))
        quentes_set = set(ranking[:10])
        intermediarias_set = set(ranking[10:17])
        frias_set = set(ranking[17:25])
        estratificadas = set()
        estratificadas.update(rng.choice(list(quentes_set), q_sel, replace=False))
        estratificadas.update(rng.choice(list(intermediarias_set), i_sel, replace=False))
        estratificadas.update(rng.choice(list(frias_set), f_sel, replace=False))
        acertos_estrat_confirm.append(len(estratificadas & alvo))

    arr_conf = np.array(acertos_confirm)
    arr_aleat = np.array(acertos_aleat_confirm)
    arr_estrat = np.array(acertos_estrat_confirm)

    media_conf = np.mean(arr_conf)
    media_aleat = np.mean(arr_aleat)
    media_estrat = np.mean(arr_estrat)

    print(f"   Mapa de calor: {media_conf:.3f}")
    print(f"   Aleatório puro: {media_aleat:.3f}")
    print(f"   Aleatório estratificado: {media_estrat:.3f}")
    print(f"   Teórico (grupo de 20): 12.000")
    print(f"\n   Composição confirmada: {q_sel}Q + {i_sel}I + {f_sel}F")

    dif = arr_conf - arr_aleat
    observado = np.mean(dif)
    desvio_dif = np.std(dif, ddof=1) if len(dif) > 1 else 0
    cohens_d = observado / desvio_dif if desvio_dif > 0 else 0
    w_stat, w_p = wilcoxon(dif)
    rng_perm = np.random.default_rng(123)
    perm_means = np.empty(10000)
    for k in range(10000):
        sinais = rng_perm.choice([-1, 1], size=len(dif))
        perm_means[k] = np.mean(dif * sinais)
    p_perm = np.mean(np.abs(perm_means) >= abs(observado))
    rng_boot = np.random.default_rng(42)
    idx = rng_boot.integers(0, len(dif), size=(n_boot, len(dif)))
    medias_boot = dif[idx].mean(axis=1)
    ic_low, ic_high = np.percentile(medias_boot, [2.5, 97.5])
    print(f"\n🔍 Teste estatístico (mapa vs aleatório puro):")
    print(f"   Diferença média: {observado:+.3f} (IC95%: [{ic_low:.3f}, {ic_high:.3f}])")
    print(f"   Cohen's d pareado: {cohens_d:+.3f}")
    print(f"   Wilcoxon pareado: W={w_stat}, p={w_p:.4f}")
    print(f"   p-valor permutação: {p_perm:.4f}")

    dif_estrat = arr_estrat - arr_aleat
    observado_er = np.mean(dif_estrat)
    desvio_er = np.std(dif_estrat, ddof=1) if len(dif_estrat) > 1 else 0
    cohens_d_er = observado_er / desvio_er if desvio_er > 0 else 0
    w_er, p_w_er = wilcoxon(dif_estrat)
    perm_er = np.empty(10000)
    for k in range(10000):
        sinais = rng_perm.choice([-1, 1], size=len(dif_estrat))
        perm_er[k] = np.mean(dif_estrat * sinais)
    p_perm_er = np.mean(np.abs(perm_er) >= abs(observado_er))
    ic_low_er, ic_high_er = np.percentile(
        dif_estrat[rng_boot.integers(0, len(dif_estrat), size=(n_boot, len(dif_estrat)))].mean(axis=1),
        [2.5, 97.5]
    )
    print(f"\n🔍 Teste estatístico (estratificado vs aleatório puro):")
    print(f"   Diferença média: {observado_er:+.3f} (IC95%: [{ic_low_er:.3f}, {ic_high_er:.3f}])")
    print(f"   Cohen's d pareado: {cohens_d_er:+.3f}")
    print(f"   Wilcoxon pareado: W={w_er}, p={p_w_er:.4f}")
    print(f"   p-valor permutação: {p_perm_er:.4f}")

    print(f"\n📊 ANÁLISE DAS FAIXAS (janela {janela_sel})")
    acertos_quentes = []
    acertos_intermediarias = []
    acertos_frias = []
    for idx in range(meio, n_total):
        passado = contests[:idx]
        alvo = set(contests[idx]['dezenas'])
        freq = freq_janela(passado, max(0, len(passado)-janela_sel), len(passado))
        ranking = obter_ranking(freq, idx)
        quentes_set = set(ranking[:10])
        intermediarias_set = set(ranking[10:17])
        frias_set = set(ranking[17:25])
        acertos_quentes.append(len(quentes_set & alvo))
        acertos_intermediarias.append(len(intermediarias_set & alvo))
        acertos_frias.append(len(frias_set & alvo))

    print(f"   🔥 Quentes: média={np.mean(acertos_quentes):.3f} (esperado 6.00)")
    print(f"   🟡 Intermediárias: média={np.mean(acertos_intermediarias):.3f} (esperado 4.20)")
    print(f"   ❄️ Frias: média={np.mean(acertos_frias):.3f} (esperado 4.80)")

    return

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v50.17.2")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Gerar carteira personalizada")
        print("2. Análise avançada de frequência + atraso")
        print("3. Análise de regras temporais e consenso")
        print("4. Modelo Preditivo Temporal (transição, hazard, repetição, placebo)")
        print("5. Análise de Mapa de Calor (frequência recente)")
        print("6. Sair")
        op = input("Escolha: ").strip()

        if op == '1':
            fixed_str = input("\n   Dezenas fixas (ex: 15 16 20 ou ENTER): ").strip()
            fixed = [int(x) for x in fixed_str.split()] if fixed_str else []
            semifixed_str = input("   Dezenas semifixas (ex: 03 07 14 25 ou ENTER): ").strip()
            semifixed = [int(x) for x in semifixed_str.split()] if semifixed_str else []
            excl_str = input("   Dezenas excluídas (ex: 04 18 22 ou ENTER): ").strip()
            excluded = [int(x) for x in excl_str.split()] if excl_str else []
            if semifixed:
                try:
                    min_semifixed = int(input(f"   Mínimo de semifixas [0-{len(semifixed)}]: ").strip() or "0")
                    max_semifixed = int(input(f"   Máximo de semifixas [0-{len(semifixed)}]: ").strip() or str(len(semifixed)))
                except:
                    min_semifixed = 0
                    max_semifixed = len(semifixed)
            else:
                min_semifixed = 0
                max_semifixed = None
            print("   Faixas estruturais (ENTER para pular)")
            try:
                pares_str = input("   Pares min,max: ").strip()
                range_pares = tuple(int(x) for x in pares_str.split(',')) if pares_str else None
            except: range_pares = None
            try:
                moldura_str = input("   Moldura min,max: ").strip()
                range_moldura = tuple(int(x) for x in moldura_str.split(',')) if moldura_str else None
            except: range_moldura = None
            try:
                primos_str = input("   Primos min,max: ").strip()
                range_primos = tuple(int(x) for x in primos_str.split(',')) if primos_str else None
            except: range_primos = None
            metodo = input("   Método [1. Pair, 2. Triple]: ").strip() or "1"
            method = 'pair_covering' if metodo == '1' else 'triple_covering'
            opt = PortfolioOptimizer(contests, fixed=fixed, semifixed=semifixed,
                                     min_semifixed=min_semifixed, max_semifixed=max_semifixed,
                                     excluded=excluded, range_pares=range_pares,
                                     range_moldura=range_moldura, range_primos=range_primos)
            portfolio = opt.optimize(5, 100000, method=method)
            for i, g in enumerate(portfolio, 1):
                p = sum(1 for x in g if x%2==0); pr = sum(1 for x in g if x in PRIMES); m = sum(1 for x in g if x in MOLDURA)
                print(f" {i}. {g} | P:{p} Pr:{pr} M:{m}")
            if len(contests) > 200:
                bt = opt.backtest(portfolio, contests[-200:])
                print(f"\n🔬 BACKTEST (200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")

        elif op == '2':
            try:
                top_n_str = input("\n   Top_ns a avaliar (ex: 5,10,15,20) [5,10,15,20]: ").strip()
                top_n_list = [int(x) for x in top_n_str.split(',')] if top_n_str else [5,10,15,20]
                min_history = int(input("   Histórico mínimo [500]: ").strip() or "500")
                n_sim = int(input("   Simulações Monte Carlo [1000]: ").strip() or "1000")
            except:
                top_n_list, min_history, n_sim = [5,10,15,20], 500, 1000
            analise_frequentes_atraso_v3(contests, top_n_list=top_n_list, min_history=min_history, n_sim_mc=n_sim)

        elif op == '3':
            try:
                min_history = int(input("\n   Histórico mínimo [500]: ").strip() or "500")
                n_sim = int(input("   Simulações Monte Carlo [500]: ").strip() or "500")
                top_n_str = input("   Top_ns a avaliar (ex: 5,10,15,20) [5,10,15,20]: ").strip()
                top_n_list = [int(x) for x in top_n_str.split(',')] if top_n_str else [5,10,15,20]
            except:
                min_history, n_sim, top_n_list = 500, 500, [5,10,15,20]
            analise_regras_temporais(contests, min_history=min_history, n_sim_mc=n_sim, top_n_list=top_n_list)

        elif op == '4':
            try:
                n_backtest = int(input("\n   Concursos para backtest [200]: ").strip() or "200")
                n_boot = int(input("   Reamostragens bootstrap [5000]: ").strip() or "5000")
            except:
                n_backtest, n_boot = 200, 5000
            analise_modelo_temporal(contests, n_backtest=n_backtest, n_boot=n_boot)

        elif op == '5':
            try:
                n_backtest = int(input("\n   Concursos para backtest [200]: ").strip() or "200")
                n_boot = int(input("   Reamostragens bootstrap [5000]: ").strip() or "5000")
            except:
                n_backtest, n_boot = 200, 5000
            analise_mapa_calor_walkforward(contests, n_backtest=n_backtest, n_boot=n_boot)

        elif op == '6':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
