#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v49.6
APENAS OPÇÕES 1 E 13

CORREÇÕES CRÍTICAS:
✅ Frequência histórica agora usa exatamente 'janela_historica' concursos (não todo o histórico)
✅ Teste de significância no treino usa teste t de uma amostra (não binomtest)
✅ Seleção de top_n com correção FDR (Benjamini-Hochberg) sobre p-valores de Wilcoxon
✅ Backtest com probabilidade teórica estimada por simulação Monte Carlo da carteira
✅ Mantém Monte Carlo vetorizado, distribuição de acertos, IC, Cohen's d, permutação
"""

import numpy as np
from scipy.stats import hypergeom, binomtest, ttest_1samp, ttest_rel, wilcoxon, mannwhitneyu, norm
from collections import defaultdict, Counter
from itertools import combinations
import os, random, time, warnings
from math import comb
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
        key = tuple(game) if isinstance(game, list) else game
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
        
        proibidas = fixed_set | semifixed_set | excluded_set
        
        todas = set(range(1, 26))
        restantes = list(todas - proibidas)
        
        n_fixas = len(fixed_set)
        
        if max_semifixed is None:
            max_semi = len(semifixed_set)
        else:
            max_semi = min(max_semifixed, len(semifixed_set))
        
        min_semi = max(min_semifixed, 0)
        if min_semi > max_semi:
            return None
        
        n_semifixed_escolher = random.randint(min_semi, max_semi)
        n_restantes = 15 - n_fixas - n_semifixed_escolher
        
        if n_restantes < 0 or n_restantes > len(restantes):
            return None
        
        for _ in range(200):
            if n_semifixed_escolher > 0 and len(semifixed_set) > 0:
                chosen_semi = set(random.sample(list(semifixed_set), min(n_semifixed_escolher, len(semifixed_set))))
            else:
                chosen_semi = set()
            
            if n_restantes > 0:
                chosen_rest = set(random.sample(restantes, min(n_restantes, len(restantes))))
            else:
                chosen_rest = set()
            
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
        return sorted(np.random.choice(range(1, 26), 15, replace=False))

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
            best_idx = -1
            best_new = -1
            
            for i, c in enumerate(candidates):
                if c in selected:
                    continue
                groups = set(combinations(sorted(c), r))
                new_groups = len(groups - covered)
                if new_groups > best_new:
                    best_new = new_groups
                    best_idx = i
            
            if best_idx == -1:
                break
            
            selected.append(candidates[best_idx])
            covered.update(combinations(sorted(candidates[best_idx]), r))
        
        return selected

    def optimize(self, n_games=5, n_candidates=100000, method='pair_covering'):
        print(f"\n🧩 CARTEIRA: {n_games} jogos | método: {method}")
        if self.fixed: print(f"   Fixas: {self.fixed}")
        if self.semifixed: print( f"   Semifixas: {self.semifixed} "
                                 f"(mín={self.min_semifixed}, máx={self.max_semifixed})" )
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
        """
        Backtest corrigido:
        empirical = frequência de concursos onde PELO MENOS uma aposta acerta 11+.
        theoretical = probabilidade estimada por simulação Monte Carlo da carteira.
        """
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

        # Estimativa teórica por simulação Monte Carlo
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
# FUNÇÕES AUXILIARES PARA ANÁLISE AVANÇADA (opção 13)
# ============================================================
def freq_janela(contests, inicio, fim, dezenas=range(1,26)):
    """Frequência de cada dezena no intervalo [inicio, fim)."""
    freq = Counter()
    inicio = max(0, inicio)
    for c in contests[inicio:fim]:
        freq.update(c['dezenas'])
    return {d: freq.get(d, 0) for d in dezenas}

def freq_janela_historica(contests, indice, janela_historica=100):
    """Frequência nos últimos 'janela_historica' concursos antes do índice."""
    inicio = max(0, indice - janela_historica)
    return freq_janela(contests, inicio, indice)

def calcular_atrasos(contests, indice=None):
    """Retorna dict de atrasos de cada dezena até o concurso 'indice' (ou final se None)."""
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

def score_dezena(freq_recente, freq_historica, atraso_z, janela_recente=10, janela_historica=100,
                 pesos=(0.5,0.2,0.3), metodo='linear'):
    """
    Calcula score combinado de uma dezena.
    freq_recente: frequência nos últimos 'janela_recente' concursos
    freq_historica: frequência nos últimos 'janela_historica' concursos
    atraso_z: z-score do atraso
    pesos: (w_recente, w_historico, w_atraso)
    metodo: 'linear', 'sqrt' ou 'log'
    """
    p = 15/25  # 0.6
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
    else:  # linear
        bonus_atraso = atraso_z

    return pesos[0] * z_recente + pesos[1] * z_hist + pesos[2] * bonus_atraso

# ============================================================
# FUNÇÃO PRINCIPAL DA OPÇÃO 13 (v49.6)
# ============================================================
def analise_frequentes_atraso_v3(contests, top_n_list=[5,10,15,20],
                                 janelas_recentes=[3,5,7,10,15,20,30,50,100],
                                 janela_historica=100, min_history=500,
                                 pesos_grid=None, n_sim_mc=1000, alpha=0.05):
    """
    Análise avançada com múltiplos top_n, métricas completas e Monte Carlo vetorizado.
    """
    print(f"\n🔬 ANÁLISE AVANÇADA DE FREQUÊNCIA + ATRASO (v49.6)")
    print(f"   Top_n avaliados: {top_n_list} | Janelas recentes: {janelas_recentes} | Janela histórica: {janela_historica}")
    print(f"   Histórico mínimo: {min_history} | Simulações MC: {n_sim_mc}")

    # Divisão treino/validação/teste
    n = len(contests)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    treino = contests[:train_end]
    validacao = contests[train_end:val_end]
    teste_oos = contests[val_end:]

    if pesos_grid is None:
        pesos_grid = [
            (0.5,0.2,0.3), (0.6,0.2,0.2), (0.4,0.3,0.3),
            (0.7,0.1,0.2), (0.3,0.3,0.4)
        ]
    metodos = ['linear', 'sqrt', 'log']

    # Estrutura para armazenar resultados de cada top_n
    resultados_topn = {}

    for top_n in top_n_list:
        print(f"\n{'='*60}")
        print(f"   🎯 TOP {top_n} DEZENAS")
        print(f"{'='*60}")

        # ---------- Busca de configuração no TREINO ----------
        melhor_score_treino = -1
        melhor_config = None
        pvals_configs = []  # para correção FDR

        for janela in janelas_recentes:
            for pesos in pesos_grid:
                for metodo in metodos:
                    acertos = []
                    for i in range(min_history, len(treino)):
                        passado = treino[:i]
                        alvo = set(treino[i]['dezenas'])
                        inicio_jan = max(0, len(passado)-janela)
                        freq = freq_janela(passado, inicio_jan, len(passado))
                        hist = freq_janela_historica(passado, len(passado), janela_historica)  # CORRIGIDO
                        atr = calcular_atrasos(passado, indice=len(passado))
                        atr_vals = np.array(list(atr.values()))
                        atr_mean, atr_std = np.mean(atr_vals), np.std(atr_vals)
                        atr_z = {d: (atr[d]-atr_mean)/(atr_std+1e-9) for d in range(1,26)}
                        scores = {d: score_dezena(freq[d], hist[d], atr_z[d], janela, janela_historica, pesos, metodo) for d in range(1,26)}
                        top = set(sorted(range(1,26), key=lambda d: scores[d], reverse=True)[:top_n])
                        acertos.append(len(top & alvo))
                    media = np.mean(acertos) if acertos else 0
                    # Teste t de uma amostra contra a média esperada 0.6*top_n
                    t_stat, p_val = ttest_1samp(acertos, 0.6*top_n) if len(acertos)>1 else (0,1.0)
                    pvals_configs.append(p_val)
                    if media > melhor_score_treino:
                        melhor_score_treino = media
                        melhor_config = (janela, pesos, metodo)

        # Correção FDR (Benjamini-Hochberg) para as configurações testadas
        pvals = np.array(pvals_configs)
        sorted_idx = np.argsort(pvals)
        m = len(pvals)
        qvals = np.ones(m)
        for i in range(m-1, -1, -1):
            rank = i + 1
            q = pvals[sorted_idx[i]] * m / rank
            qvals[sorted_idx[i]] = min(q, qvals[sorted_idx[i+1]] if i < m-1 else 1.0)
        best_config_idx = np.argmin(pvals)
        if qvals[best_config_idx] < 0.2:
            print(f"   🔍 Melhor configuração significativa após FDR (q={qvals[best_config_idx]:.4f}).")
        else:
            print(f"   ⚠️ Nenhuma configuração significativa após FDR; usando melhor média absoluta.")

        janela_opt, pesos_opt, metodo_opt = melhor_config

        # ---------- Validação ----------
        acertos_val = []
        for i in range(len(validacao)):
            passado = treino + validacao[:i]
            alvo = set(validacao[i]['dezenas'])
            inicio_jan = max(0, len(passado)-janela_opt)
            freq = freq_janela(passado, inicio_jan, len(passado))
            hist = freq_janela_historica(passado, len(passado), janela_historica)  # CORRIGIDO
            atr = calcular_atrasos(passado, indice=len(passado))
            atr_vals = np.array(list(atr.values()))
            atr_mean, atr_std = np.mean(atr_vals), np.std(atr_vals)
            atr_z = {d: (atr[d]-atr_mean)/(atr_std+1e-9) for d in range(1,26)}
            scores = {d: score_dezena(freq[d], hist[d], atr_z[d], janela_opt, janela_historica, pesos_opt, metodo_opt) for d in range(1,26)}
            top = set(sorted(range(1,26), key=lambda d: scores[d], reverse=True)[:top_n])
            acertos_val.append(len(top & alvo))
        media_val = np.mean(acertos_val) if acertos_val else 0

        if media_val < melhor_score_treino - 0.5:
            print(f"   ⚠️ Degradação na validação; usando configuração padrão (janela=10, linear).")
            janela_opt, pesos_opt, metodo_opt = 10, (0.5,0.2,0.3), 'linear'

        # ---------- Teste OOS ----------
        estrategias = ['aleatorio', 'frequencia', 'atraso', 'freq+atraso_linear', 'freq+atraso_sqrt', 'freq+atraso_log', 'modelo_otimizado']
        resultados_oos = {est: [] for est in estrategias}

        for i in range(len(teste_oos)):
            passado = treino + validacao + teste_oos[:i]
            alvo = set(teste_oos[i]['dezenas'])

            inicio_jan10 = max(0, len(passado)-10)
            freq10 = freq_janela(passado, inicio_jan10, len(passado))
            hist = freq_janela_historica(passado, len(passado), janela_historica)  # CORRIGIDO
            atr = calcular_atrasos(passado, indice=len(passado))
            atr_vals = np.array(list(atr.values()))
            atr_mean, atr_std = np.mean(atr_vals), np.std(atr_vals)
            atr_z = {d: (atr[d]-atr_mean)/(atr_std+1e-9) for d in range(1,26)}

            # Aleatório
            selecionadas = set(np.random.choice(range(1,26), top_n, replace=False))
            resultados_oos['aleatorio'].append(len(selecionadas & alvo))

            # Frequência
            top_freq = set(sorted(range(1,26), key=lambda d: (freq10[d], hist[d]), reverse=True)[:top_n])
            resultados_oos['frequencia'].append(len(top_freq & alvo))

            # Atraso
            top_atr = set(sorted(range(1,26), key=lambda d: atr[d], reverse=True)[:top_n])
            resultados_oos['atraso'].append(len(top_atr & alvo))

            # Combinações
            for metodo, nome in [('linear','freq+atraso_linear'), ('sqrt','freq+atraso_sqrt'), ('log','freq+atraso_log')]:
                scores = {d: score_dezena(freq10[d], hist[d], atr_z[d], 10, janela_historica, (0.5,0.2,0.3), metodo) for d in range(1,26)}
                top = set(sorted(range(1,26), key=lambda d: scores[d], reverse=True)[:top_n])
                resultados_oos[nome].append(len(top & alvo))

            # Modelo otimizado
            inicio_jan_opt = max(0, len(passado)-janela_opt)
            freq_jan = freq_janela(passado, inicio_jan_opt, len(passado))
            scores = {d: score_dezena(freq_jan[d], hist[d], atr_z[d], janela_opt, janela_historica, pesos_opt, metodo_opt) for d in range(1,26)}
            top = set(sorted(range(1,26), key=lambda d: scores[d], reverse=True)[:top_n])
            resultados_oos['modelo_otimizado'].append(len(top & alvo))

        # ---------- Métricas detalhadas ----------
        print(f"\n📊 Comparação OOS (top {top_n}):")
        print(f"{'Estratégia':<20} {'Média':<8} {'≥7':<8} {'≥8':<8} {'≥9':<8} {'≥10':<8}")
        print("-"*60)
        arrs = {}
        for est in estrategias:
            arr = np.array(resultados_oos[est])
            arrs[est] = arr
            media = np.mean(arr)
            p7 = np.mean(arr >= 7)*100
            p8 = np.mean(arr >= 8)*100
            p9 = np.mean(arr >= 9)*100
            p10 = np.mean(arr >= 10)*100
            print(f"{est:<20} {media:<8.3f} {p7:<8.1f} {p8:<8.1f} {p9:<8.1f} {p10:<8.1f}")

        # Diferença modelo vs aleatório
        dif = arrs['modelo_otimizado'] - arrs['aleatorio']
        media_dif = np.mean(dif)
        # IC bootstrap
        rng = np.random.default_rng(42)
        boot_means = [np.mean(rng.choice(dif, size=len(dif), replace=True)) for _ in range(1000)]
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        # Cohen's d
        pooled_std = np.sqrt((np.std(arrs['modelo_otimizado'])**2 + np.std(arrs['aleatorio'])**2)/2)
        cohens_d = media_dif / pooled_std if pooled_std > 0 else 0
        # Teste de permutação
        perm_diffs = []
        combined = np.concatenate([arrs['modelo_otimizado'], arrs['aleatorio']])
        for _ in range(1000):
            perm = rng.permutation(combined)
            perm_diffs.append(np.mean(perm[:len(arrs['modelo_otimizado'])]) - np.mean(perm[len(arrs['modelo_otimizado']):]))
        p_perm = np.mean(np.abs(perm_diffs) >= abs(media_dif))

        print(f"\n🔍 Comparação modelo vs aleatório:")
        print(f"   Diferença média: {media_dif:.3f} (IC95%: [{ci_low:.3f}, {ci_high:.3f}])")
        print(f"   Cohen's d: {cohens_d:.3f}")
        print(f"   p-valor (permutação): {p_perm:.4f}")

        # Teste Wilcoxon
        w, p_wilc = wilcoxon(arrs['modelo_otimizado'], arrs['aleatorio'])
        print(f"   Wilcoxon: W={w}, p={p_wilc:.4f}")

        # Guarda resultados para este top_n
        resultados_topn[top_n] = {
            'arrs': arrs,
            'media_dif': media_dif,
            'cohens_d': cohens_d,
            'p_perm': p_perm,
            'p_wilc': p_wilc,
            'melhor_config': melhor_config
        }

    # ---------- Correção FDR sobre os top_n ----------
    pvals_topn = [resultados_topn[tn]['p_wilc'] for tn in top_n_list]
    # Aplicar Benjamini-Hochberg manualmente
    m = len(pvals_topn)
    sorted_idx = np.argsort(pvals_topn)
    qvals = np.ones(m)
    for i in range(m-1, -1, -1):
        rank = i + 1
        q = pvals_topn[sorted_idx[i]] * m / rank
        qvals[sorted_idx[i]] = min(q, qvals[sorted_idx[i+1]] if i < m-1 else 1.0)
    print(f"\n🔍 Correção FDR sobre os top_n:")
    for j, tn in enumerate(top_n_list):
        print(f"   Top {tn}: p-valor Wilcoxon = {pvals_topn[j]:.4f} | q = {qvals[j]:.4f}")

    # Selecionar top_n para MC: usar o que tiver menor q (se significativo), senão 10
    melhor_topn_idx = np.argmin(qvals)
    melhor_topn = top_n_list[melhor_topn_idx]
    if qvals[melhor_topn_idx] >= alpha:
        print(f"   Nenhum top_n significativo; usando top 10 para Monte Carlo.")
        melhor_topn = 10
    else:
        print(f"   Top {melhor_topn} selecionado para Monte Carlo.")

    # ---------- Monte Carlo vetorizado ----------
    print(f"\n🎲 Monte Carlo VETORIZADO ({n_sim_mc} simulações) para top {melhor_topn}")
    res_mc = resultados_topn[melhor_topn]
    obs_media = np.mean(res_mc['arrs']['modelo_otimizado'])

    def preparar_features_mc(sim_contests, janelas):
        n_sim = len(sim_contests)
        presenca = np.zeros((n_sim, 25), dtype=np.int8)
        for i, c in enumerate(sim_contests):
            presenca[i, np.array(c['dezenas']) - 1] = 1

        cum = np.zeros((n_sim + 1, 25), dtype=np.int16)
        np.cumsum(presenca, axis=0, out=cum[1:])

        atrasos = np.zeros((n_sim + 1, 25), dtype=np.int16)
        ultimo = np.full(25, -1, dtype=np.int32)
        for i in range(n_sim + 1):
            if i > 0:
                atrasos[i] = np.where(ultimo >= 0, i - 1 - ultimo, i)
                dezenas = np.flatnonzero(presenca[i - 1])
                ultimo[dezenas] = i - 1

        atr_mean = atrasos.mean(axis=1)
        atr_std = atrasos.std(axis=1)
        atr_z = (atrasos - atr_mean[:, None]) / (atr_std[:, None] + 1e-9)

        freq_recent = {}
        for janela in janelas:
            inicio = np.maximum(np.arange(n_sim + 1) - janela, 0)
            freq_recent[janela] = cum - cum[inicio]

        return cum, freq_recent, atr_z

    def avaliar_configuracoes_mc(cum, freq_recent, atr_z, indices, alvos, janelas,
                                 pesos_grid, metodos, top_n, janela_historica):
        configs = []
        for janela in janelas:
            for pesos in pesos_grid:
                for metodo in metodos:
                    configs.append((janela, pesos, metodo))

        n_cfg = len(configs)
        acertos = np.zeros((n_cfg, len(indices)), dtype=np.float32)

        for ci, (janela, pesos, metodo) in enumerate(configs):
            w_rec, w_hist, w_atr = pesos
            freq = freq_recent[janela][indices]
            # CORREÇÃO: histórico limitado a janela_historica
            inicio_hist = np.maximum(indices - janela_historica, 0)
            hist = cum[indices] - cum[inicio_hist]

            p = 15 / 25
            media_rec = p * janela
            desvio_rec = np.sqrt(janela * p * (1 - p))
            z_rec = (freq - media_rec) / desvio_rec

            media_hist = p * janela_historica
            desvio_hist = np.sqrt(janela_historica * p * (1 - p))
            z_hist = (hist - media_hist) / desvio_hist

            z_atr = atr_z[indices]
            if metodo == 'sqrt':
                bonus_atr = np.sqrt(np.maximum(z_atr, 0)) * np.sign(z_atr)
            elif metodo == 'log':
                bonus_atr = np.log1p(np.maximum(z_atr, 0)) * np.sign(z_atr)
            else:
                bonus_atr = z_atr

            scores = w_rec * z_rec + w_hist * z_hist + w_atr * bonus_atr

            top_idx = np.argpartition(scores, -top_n, axis=1)[:, -top_n:]
            top_dezenas = top_idx + 1

            for j, alvo in enumerate(alvos):
                acertos[ci, j] = np.isin(top_dezenas[j], list(alvo)).sum()

        medias = acertos.mean(axis=1)
        melhor_idx = np.argmax(medias)
        return medias[melhor_idx], configs[melhor_idx]

    medias_sim = np.empty(n_sim_mc, dtype=np.float64)

    for sim in range(n_sim_mc):
        matriz_sim = np.empty((n, 15), dtype=np.int8)
        for i in range(n):
            matriz_sim[i] = np.random.choice(np.arange(1, 26), 15, replace=False)
        sim_contests = [{'dezenas': np.sort(matriz_sim[i]).tolist()} for i in range(n)]

        cum, freq_recent, atr_z = preparar_features_mc(sim_contests, janelas_recentes)

        train_indices = np.arange(min_history, train_end)
        train_targets = [set(matriz_sim[i]) for i in train_indices]
        best_score, best_cfg = avaliar_configuracoes_mc(
            cum=cum, freq_recent=freq_recent, atr_z=atr_z,
            indices=train_indices, alvos=train_targets,
            janelas=janelas_recentes, pesos_grid=pesos_grid,
            metodos=metodos, top_n=melhor_topn, janela_historica=janela_historica
        )
        janela_sim, pesos_sim, metodo_sim = best_cfg

        val_indices = np.arange(train_end, val_end)
        val_targets = [set(matriz_sim[i]) for i in val_indices]
        w_rec, w_hist, w_atr = pesos_sim

        freq = freq_recent[janela_sim][val_indices]
        inicio_hist = np.maximum(val_indices - janela_historica, 0)
        hist = cum[val_indices] - cum[inicio_hist]
        p = 15 / 25
        media_rec = p * janela_sim
        desvio_rec = np.sqrt(janela_sim * p * (1 - p))
        z_rec = (freq - media_rec) / desvio_rec
        media_hist = p * janela_historica
        desvio_hist = np.sqrt(janela_historica * p * (1 - p))
        z_hist = (hist - media_hist) / desvio_hist
        z_atr = atr_z[val_indices]

        if metodo_sim == 'sqrt':
            bonus_atr = np.sqrt(np.maximum(z_atr, 0)) * np.sign(z_atr)
        elif metodo_sim == 'log':
            bonus_atr = np.log1p(np.maximum(z_atr, 0)) * np.sign(z_atr)
        else:
            bonus_atr = z_atr

        scores = w_rec * z_rec + w_hist * z_hist + w_atr * bonus_atr
        top_idx = np.argpartition(scores, -melhor_topn, axis=1)[:, -melhor_topn:]
        top_dezenas = top_idx + 1
        acertos_val = np.array([np.isin(top_dezenas[j], list(val_targets[j])).sum() for j in range(len(val_indices))])
        media_val_sim = acertos_val.mean()

        if media_val_sim < best_score - 0.5:
            janela_sim, pesos_sim, metodo_sim = 10, (0.5,0.2,0.3), 'linear'

        test_indices = np.arange(val_end, n)
        test_targets = [set(matriz_sim[i]) for i in test_indices]
        w_rec, w_hist, w_atr = pesos_sim

        freq = freq_recent[janela_sim][test_indices]
        inicio_hist = np.maximum(test_indices - janela_historica, 0)
        hist = cum[test_indices] - cum[inicio_hist]
        media_rec = p * janela_sim
        desvio_rec = np.sqrt(janela_sim * p * (1 - p))
        z_rec = (freq - media_rec) / desvio_rec
        z_hist = (hist - media_hist) / desvio_hist
        z_atr = atr_z[test_indices]

        if metodo_sim == 'sqrt':
            bonus_atr = np.sqrt(np.maximum(z_atr, 0)) * np.sign(z_atr)
        elif metodo_sim == 'log':
            bonus_atr = np.log1p(np.maximum(z_atr, 0)) * np.sign(z_atr)
        else:
            bonus_atr = z_atr

        scores = w_rec * z_rec + w_hist * z_hist + w_atr * bonus_atr
        top_idx = np.argpartition(scores, -melhor_topn, axis=1)[:, -melhor_topn:]
        top_dezenas = top_idx + 1
        acertos_oos = np.array([np.isin(top_dezenas[j], list(test_targets[j])).sum() for j in range(len(test_indices))])
        medias_sim[sim] = acertos_oos.mean()

        if (sim + 1) % max(1, n_sim_mc // 10) == 0 or sim == 0:
            print(f"   Simulação {sim+1}/{n_sim_mc} | Média atual: {medias_sim[:sim+1].mean():.4f} | Config: janela={janela_sim}, pesos={pesos_sim}, método={metodo_sim}")

    medias_sim = np.asarray(medias_sim)
    p_mc = np.mean(medias_sim >= obs_media)

    print("\n" + "=" * 60)
    print("🎲 RESULTADO DO MONTE CARLO")
    print("=" * 60)
    print(f"Média observada (modelo otimizado): {obs_media:.3f}")
    print(f"Média sob sorteios independentes: {np.mean(medias_sim):.3f} ± {np.std(medias_sim):.3f}")
    print(f"Mediana das simulações: {np.median(medias_sim):.3f}")
    print(f"Melhor simulação: {np.max(medias_sim):.3f}")
    print(f"p-valor empírico: {p_mc:.4f} {'🔍' if p_mc < alpha else ''}")
    q025, q975 = np.percentile(medias_sim, [2.5, 97.5])
    print(f"IC empírico 95% das médias simuladas: [{q025:.3f}, {q975:.3f}]")

    # Geração de carteira final com o melhor modelo
    print(f"\n🧩 Gerando carteira otimizada com as top {melhor_topn} dezenas do melhor modelo...")
    janela_final, pesos_final, metodo_final = res_mc['melhor_config']
    passado_total = contests
    inicio_jan = max(0, len(passado_total)-janela_final)
    freq = freq_janela(passado_total, inicio_jan, len(passado_total))
    hist = freq_janela_historica(passado_total, len(passado_total), janela_historica)
    atr = calcular_atrasos(passado_total)
    atr_vals = np.array(list(atr.values()))
    atr_mean, atr_std = np.mean(atr_vals), np.std(atr_vals)
    atr_z = {d: (atr[d]-atr_mean)/(atr_std+1e-9) for d in range(1,26)}
    scores = {d: score_dezena(freq[d], hist[d], atr_z[d], janela_final, janela_historica, pesos_final, metodo_final) for d in range(1,26)}
    top_final = sorted(range(1,26), key=lambda d: scores[d], reverse=True)[:melhor_topn]

    print(f"   Top {melhor_topn} dezenas: {top_final}")

    try:
        opt = PortfolioOptimizer(contests, fixed=top_final)
        portfolio = opt.optimize(5, 50000, method='pair_covering')
        for i, g in enumerate(portfolio, 1):
            p = sum(1 for x in g if x%2==0)
            pr = sum(1 for x in g if x in PRIMES)
            m = sum(1 for x in g if x in MOLDURA)
            print(f" {i}. {g} | P:{p} Pr:{pr} M:{m}")
        if len(contests) > 200:
            bt = opt.backtest(portfolio, contests[-200:])
            print(f"\n🔬 BACKTEST (200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
    except Exception as e:
        print(f"   Erro ao gerar carteira: {e}")

    return resultados_topn, p_mc

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v49.6")
    print("   OPÇÕES 1 E 13")
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
        print("13. Análise avançada de frequência + atraso (v49.6)")
        print("0. Sair")
        op = input("Escolha: ").strip()
        
        if op == '1':
            fixed_str = input( "\n   Dezenas fixas (ex: 15 16 20 ou ENTER): ").strip()  
            fixed = [int(x) for x in fixed_str.split()] if fixed_str else []
              
            semifixed_str = input(
                "   Dezenas semifixas (ex: 03 07 14 25 ou ENTER): ").strip()
            semifixed = [int(x) for x in semifixed_str.split()] if semifixed_str else []
            
            excl_str = input("   Dezenas excluídas (ex: 04 18 22 ou ENTER): ").strip()
            excluded = [int(x) for x in excl_str.split()] if excl_str else []
            
            if semifixed:
                try:
                    min_semifixed = int(
                        input( f"   Mínimo de semifixas [0-{len(semifixed)}]: " ).strip() or "0" )
                    max_semifixed = int( input( f"   Máximo de semifixas [0-{len(semifixed)}]: " ).strip() or str(len(semifixed)) )
                except:
                    min_semifixed = 0
                    max_semifixed = len(semifixed)
            else:
                min_semifixed = 0
                max_semifixed = None
                
            print("   Faixas estruturais (ENTER para pular)")
            try:
                pares_str = input("   Pares min,max (ex: 7,9): ").strip()
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
            metodo = input("\n   Método [1. Pair, 2. Triple]: ").strip() or "1"
            method = 'pair_covering' if metodo == '1' else 'triple_covering'
            
            opt = PortfolioOptimizer(
                contests,
                fixed=fixed,
                semifixed=semifixed,
                min_semifixed=min_semifixed,
                max_semifixed=max_semifixed,
                excluded=excluded,
                range_pares=range_pares,
                range_moldura=range_moldura,
                range_primos=range_primos)
            portfolio = opt.optimize(5, 100000, method=method)
            for i, g in enumerate(portfolio, 1):
                p = sum(1 for x in g if x%2==0); pr = sum(1 for x in g if x in PRIMES); m = sum(1 for x in g if x in MOLDURA)
                print(f" {i}. {g} | P:{p} Pr:{pr} M:{m}")
            if len(contests) > 200:
                bt = opt.backtest(portfolio, contests[-200:])
                print(f"\n🔬 BACKTEST (200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
        
        elif op == '13':
            try:
                top_n_str = input("\n   Top_ns a avaliar (ex: 5,10,15,20) [5,10,15,20]: ").strip()
                top_n_list = [int(x) for x in top_n_str.split(',')] if top_n_str else [5,10,15,20]
                min_history = int(input("   Histórico mínimo [500]: ").strip() or "500")
                n_sim = int(input("   Simulações Monte Carlo [1000]: ").strip() or "1000")
            except:
                top_n_list, min_history, n_sim = [5,10,15,20], 500, 1000
            analise_frequentes_atraso_v3(contests, top_n_list=top_n_list, min_history=min_history, n_sim_mc=n_sim)
        
        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
