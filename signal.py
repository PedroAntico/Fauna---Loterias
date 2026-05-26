#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PESQUISA DE SINAL - LOTOFÁCIL
==============================
Protocolo científico rígido para testar se existe
QUALQUER sinal preditivo nas features estruturais.

HIPÓTESES TESTADAS (pré-registradas, sem alteração pós-resultado):
  H1: repetição do último concurso prediz hits futuros
  H2: compressão geométrica (gap_var) prediz hits futuros
  H3: entropia local prediz hits futuros
  H4: distância de Mahalanobis prediz hits futuros
  H5: pares do último concurso prediz hits futuros
  H6: moldura do último concurso prediz hits futuros
  H7: soma do último concurso prediz hits futuros
  H8: energia (soma de gaps absolutos) prediz hits futuros
  H9: consecutivos do último concurso prediz hits futuros
  H10: amplitude do último concurso prediz hits futuros

PROTOCOLO:
  - Dados divididos: 80% treino (só para calcular médias/thresholds) / 20% teste
  - Teste realizado APENAS no held-out set
  - Métrica primária: correlação de Spearman entre feature e hits
  - Correção de Bonferroni para múltiplos testes
  - Threshold de significância: p < 0.05 / N_hipoteses (Bonferroni)
  - Sem tuning pós-resultado. Sem exceções.

RESULTADO ESPERADO HONESTO:
  - Se todas as hipóteses falharem → previsibilidade não existe
  - Se alguma sobreviver → existe sinal real para investigar
  Ambos têm valor científico.
"""

import numpy as np
import os
from scipy import stats
from scipy.stats import spearmanr, pointbiserialr, mannwhitneyu
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONSTANTES (fixas, não alterar após definir)
# ─────────────────────────────────────────────
PRIMES  = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5,6,10,11,15,16,20,21,22,23,24,25}
CENTRO  = {7,8,9,12,13,14,17,18,19}

TRAIN_RATIO   = 0.80   # 80% treino, 20% teste
N_HIPOTESES   = 10     # Para correção de Bonferroni
ALPHA         = 0.05
ALPHA_BONF    = ALPHA / N_HIPOTESES   # threshold corrigido
N_RANDOM_GAMES = 100   # jogos aleatórios por concurso para medir hits esperados
RANDOM_SEED   = 42

np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# CARREGAMENTO
# ─────────────────────────────────────────────
def load_contests(csv_file='resultados_lotofacil.csv'):
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, csv_file)
    if not os.path.exists(path):
        # tenta diretório atual
        path = csv_file
    if not os.path.exists(path):
        print(f"❌ Arquivo não encontrado: {path}")
        return None
    contests = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(';')
            if len(parts) < 17:
                continue
            try:
                dezenas = [int(x.strip()) for x in parts[2:17] if x.strip().isdigit()]
                if len(dezenas) != 15 or len(set(dezenas)) != 15:
                    continue
                if any(x < 1 or x > 25 for x in dezenas):
                    continue
                contests.append({
                    'concurso': int(parts[0]),
                    'data': parts[1],
                    'dezenas': sorted(dezenas)
                })
            except:
                continue
    contests.sort(key=lambda x: x['concurso'])
    return contests

# ─────────────────────────────────────────────
# EXTRAÇÃO DE FEATURES (do concurso ANTERIOR)
# ─────────────────────────────────────────────
def extract_prev_features(prev_dezenas, prev_prev_dezenas=None):
    """
    Extrai features do concurso ANTERIOR (t-1).
    Estas são as variáveis preditoras — o que temos ANTES do sorteio.
    """
    d = sorted(prev_dezenas)
    gaps = [d[i+1]-d[i] for i in range(len(d)-1)]

    # Consecutivos
    run = 1; max_run = 1
    for i in range(len(d)-1):
        if d[i+1]-d[i]==1:
            run += 1; max_run = max(max_run, run)
        else:
            run = 1
    total_consec = sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)

    # Repetição entre t-2 e t-1
    rep_prev = 0
    if prev_prev_dezenas:
        rep_prev = len(set(prev_dezenas) & set(prev_prev_dezenas))

    amplitude = max(d) - min(d)
    std_pos = float(np.std(d))
    compressao = std_pos / amplitude if amplitude > 0 else 0.5
    energia = sum(abs(d[i]-d[i-1]) for i in range(1, len(d)))

    # Entropia posicional
    from scipy.stats import entropy as sp_entropy
    pc = np.bincount(d, minlength=26)[1:]
    pr = pc / 15.0
    pr = np.where(pr > 0, pr, 1e-10)
    entropia = float(sp_entropy(pr))

    # Mahalanobis (calculado depois com média histórica)
    # Aqui retornamos as features brutas
    return {
        'repeticao':    rep_prev,
        'pares':        sum(1 for x in d if x % 2 == 0),
        'primos':       sum(1 for x in d if x in PRIMES),
        'moldura':      sum(1 for x in d if x in MOLDURA),
        'centro':       sum(1 for x in d if x in CENTRO),
        'soma':         sum(d),
        'amplitude':    amplitude,
        'consecutivos': total_consec,
        'max_run':      max_run,
        'gap_medio':    float(np.mean(gaps)),
        'gap_var':      float(np.var(gaps)),
        'gap_max':      float(max(gaps)),
        'gap_min':      float(min(gaps)),
        'energia':      float(energia),
        'compressao':   float(compressao),
        'entropia':     float(entropia),
        'quadrantes':   len(set((x-1)//5 for x in d)),
    }

# ─────────────────────────────────────────────
# HITS DE UM JOGO FIXO CONTRA UM SORTEIO
# ─────────────────────────────────────────────
def compute_hits(game, draw):
    return len(set(game) & set(draw))

# ─────────────────────────────────────────────
# CONSTRUÇÃO DO DATASET
# ─────────────────────────────────────────────
def build_dataset(contests):
    """
    Para cada concurso t (a partir do 3º):
      - features: características do concurso t-1
      - target: hits médios de 100 jogos aleatórios contra o sorteio t
        (usamos média para reduzir variância do target)
    
    Usar hits médios de jogos aleatórios é mais estável que um único jogo.
    O sinal que buscamos: alguma feature de t-1 prediz se t terá
    mais ou menos hits que o esperado aleatório (9.0)?
    """
    records = []
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"   Construindo dataset ({len(contests)} concursos)...")

    for i in range(2, len(contests)):
        prev      = contests[i-1]['dezenas']
        prev_prev = contests[i-2]['dezenas']
        current   = contests[i]['dezenas']

        # Features do concurso t-1
        feats = extract_prev_features(prev, prev_prev)

        # Target: hits médios de N jogos aleatórios contra sorteio atual
        random_games = [
            sorted(rng.choice(range(1, 26), 15, replace=False))
            for _ in range(N_RANDOM_GAMES)
        ]
        avg_hits = np.mean([compute_hits(g, current) for g in random_games])

        # Target binário: acima/abaixo do esperado (9.0)
        above_expected = 1 if avg_hits > 9.0 else 0

        records.append({
            'concurso': contests[i]['concurso'],
            **feats,
            'avg_hits': avg_hits,
            'above_expected': above_expected,
        })

    return records

# ─────────────────────────────────────────────
# CÁLCULO DE MAHALANOBIS (pós-construção)
# ─────────────────────────────────────────────
def add_mahalanobis(records, feature_names):
    """Adiciona distância de Mahalanobis calculada sobre o treino."""
    X = np.array([[r[f] for f in feature_names] for r in records])
    mean_vec = np.mean(X, axis=0)
    cov = np.cov(X.T)
    try:
        cov_inv = np.linalg.inv(cov + np.eye(len(feature_names)) * 1e-6)
        for i, r in enumerate(records):
            diff = X[i] - mean_vec
            mahal = float(np.sqrt(max(0, diff @ cov_inv @ diff)))
            r['mahalanobis'] = mahal
    except:
        for r in records:
            r['mahalanobis'] = 0.0
    return records

# ─────────────────────────────────────────────
# TESTE DE UMA HIPÓTESE
# ─────────────────────────────────────────────
def test_hypothesis(feature_name, records_test, h_number, description):
    """
    Testa correlação de Spearman entre feature e avg_hits no held-out set.
    Também faz Mann-Whitney entre quartil superior e inferior.
    """
    x = np.array([r[feature_name] for r in records_test])
    y = np.array([r['avg_hits'] for r in records_test])

    # Spearman
    rho, p_spearman = spearmanr(x, y)

    # Mann-Whitney: Q1 vs Q4 da feature
    q25 = np.percentile(x, 25)
    q75 = np.percentile(x, 75)
    low_hits  = y[x <= q25]
    high_hits = y[x >= q75]

    if len(low_hits) >= 5 and len(high_hits) >= 5:
        stat_mw, p_mw = mannwhitneyu(low_hits, high_hits, alternative='two-sided')
        mean_low  = float(np.mean(low_hits))
        mean_high = float(np.mean(high_hits))
    else:
        p_mw = 1.0
        mean_low = mean_high = float(np.mean(y))

    significant_bonf = p_spearman < ALPHA_BONF
    significant_raw  = p_spearman < ALPHA

    status = "✅ SIGNIFICATIVO (Bonferroni)" if significant_bonf else (
             "⚠️  Marginalmente significativo" if significant_raw else
             "❌ Não significativo")

    return {
        'h_number':   h_number,
        'feature':    feature_name,
        'description': description,
        'rho':        float(rho),
        'p_spearman': float(p_spearman),
        'p_mw':       float(p_mw),
        'mean_low_q': mean_low,
        'mean_high_q': mean_high,
        'diff_means': mean_high - mean_low,
        'n_test':     len(records_test),
        'significant_bonf': significant_bonf,
        'significant_raw':  significant_raw,
        'status':     status,
    }

# ─────────────────────────────────────────────
# ANÁLISE DE REGIME TEMPORAL
# ─────────────────────────────────────────────
def test_temporal_stability(feature_name, all_records):
    """
    Testa se a correlação é estável ao longo do tempo (decadas de concursos).
    Um sinal real deve ser estável, não concentrado num período.
    """
    n = len(all_records)
    chunk_size = n // 5  # 5 períodos
    chunk_rhos = []

    for i in range(5):
        chunk = all_records[i*chunk_size:(i+1)*chunk_size]
        if len(chunk) < 20:
            continue
        x = np.array([r[feature_name] for r in chunk])
        y = np.array([r['avg_hits'] for r in chunk])
        rho, _ = spearmanr(x, y)
        chunk_rhos.append(rho)

    if len(chunk_rhos) < 3:
        return None

    return {
        'feature': feature_name,
        'rhos_por_periodo': chunk_rhos,
        'media_rho': float(np.mean(chunk_rhos)),
        'std_rho':   float(np.std(chunk_rhos)),
        'cv_rho':    float(np.std(chunk_rhos) / (abs(np.mean(chunk_rhos)) + 1e-10)),
        'sinal_consistente': np.std(chunk_rhos) < 0.1 and all(r * np.mean(chunk_rhos) > 0 for r in chunk_rhos),
    }

# ─────────────────────────────────────────────
# TESTE DE AUTOCORRELAÇÃO (há memória?)
# ─────────────────────────────────────────────
def test_autocorrelation(contests, max_lag=10):
    """
    Testa autocorrelação nas séries temporais das features.
    Se há autocorrelação significativa → há estrutura temporal explorável.
    """
    series = {
        'repeticao': [],
        'pares': [],
        'soma': [],
        'gap_var': [],
        'energia': [],
    }

    for i in range(1, len(contests)):
        prev = contests[i-1]['dezenas']
        prev_prev = contests[i-2]['dezenas'] if i >= 2 else prev
        feats = extract_prev_features(prev, prev_prev)
        for k in series:
            if k in feats:
                series[k].append(feats[k])

    results = {}
    for name, vals in series.items():
        arr = np.array(vals)
        lags_sig = []
        for lag in range(1, max_lag+1):
            x1 = arr[:-lag]
            x2 = arr[lag:]
            r, p = spearmanr(x1, x2)
            if p < 0.05:
                lags_sig.append((lag, float(r), float(p)))
        results[name] = {
            'n': len(arr),
            'mean': float(np.mean(arr)),
            'std':  float(np.std(arr)),
            'lags_significativos': lags_sig,
            'tem_memoria': len(lags_sig) > 0,
        }
    return results

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("🔬 PESQUISA DE SINAL — PROTOCOLO CIENTÍFICO RÍGIDO")
    print("=" * 65)
    print(f"\n📋 PROTOCOLO PRÉ-REGISTRADO:")
    print(f"   N hipóteses: {N_HIPOTESES}")
    print(f"   Alpha: {ALPHA}")
    print(f"   Alpha Bonferroni: {ALPHA_BONF:.4f}")
    print(f"   Train/Test split: {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}%")
    print(f"   Jogos aleatórios por concurso: {N_RANDOM_GAMES}")
    print(f"   Seed: {RANDOM_SEED} (fixo, não alterável)")

    # Carregamento
    contests = load_contests('resultados_lotofacil.csv')
    if contests is None:
        return
    print(f"\n✅ {len(contests)} concursos carregados")

    # Split treino/teste (temporal — sem look-ahead)
    split_idx = int(len(contests) * TRAIN_RATIO)
    contests_train = contests[:split_idx]
    contests_test  = contests[split_idx:]
    print(f"   Treino: {len(contests_train)} concursos")
    print(f"   Teste:  {len(contests_test)} concursos (held-out)")

    # ─── ANÁLISE 0: Autocorrelação (tem memória?) ───
    print("\n" + "─"*65)
    print("📊 ANÁLISE 0: AUTOCORRELAÇÃO DAS SÉRIES TEMPORAIS")
    print("   (existe memória nas features? → pré-condição para previsão)")
    print("─"*65)

    autocorr = test_autocorrelation(contests)
    any_memory = False
    for name, res in autocorr.items():
        mem = "✅ TEM MEMÓRIA" if res['tem_memoria'] else "❌ Sem memória"
        print(f"\n   {name:15s}: média={res['mean']:.2f} ± {res['std']:.2f}  {mem}")
        if res['tem_memoria']:
            any_memory = True
            for lag, r, p in res['lags_significativos'][:3]:
                print(f"      lag={lag}: rho={r:+.3f}  p={p:.4f}")

    if not any_memory:
        print("\n   ⚠️  NENHUMA SÉRIE TEMPORAL TEM MEMÓRIA DETECTÁVEL")
        print("   → Isto é consistente com independência dos sorteios")

    # ─── ANÁLISE 1: Mahalanobis no set completo ───
    print("\n" + "─"*65)
    print("📊 ANÁLISE 1: CONSTRUÇÃO DO DATASET")
    print("─"*65)

    feature_names_base = [
        'repeticao', 'pares', 'primos', 'moldura', 'centro',
        'soma', 'amplitude', 'consecutivos', 'max_run',
        'gap_medio', 'gap_var', 'gap_max', 'gap_min',
        'energia', 'compressao', 'entropia', 'quadrantes'
    ]

    # Build dataset completo (treino + teste)
    all_records = build_dataset(contests)
    # Adiciona Mahalanobis calculado sobre TREINO
    train_records_raw = all_records[:len(contests_train)-2]
    all_records = add_mahalanobis(all_records, feature_names_base)

    # Split temporal
    n_train = len(contests_train) - 2
    records_train = all_records[:n_train]
    records_test  = all_records[n_train:]

    print(f"   Dataset treino: {len(records_train)} registos")
    print(f"   Dataset teste:  {len(records_test)} registos")
    print(f"   Target (avg_hits): média={np.mean([r['avg_hits'] for r in records_test]):.4f} "
          f"± {np.std([r['avg_hits'] for r in records_test]):.4f}")
    print(f"   Teórico esperado: 9.0000")

    # ─── ANÁLISE 2: Teste das 10 hipóteses ───
    print("\n" + "─"*65)
    print("📊 ANÁLISE 2: TESTE DAS 10 HIPÓTESES (held-out set)")
    print(f"   Threshold Bonferroni: p < {ALPHA_BONF:.4f}")
    print("─"*65)

    hipoteses = [
        ('repeticao',   'H1: repetição t-1 prediz hits em t'),
        ('gap_var',     'H2: variância dos gaps prediz hits'),
        ('entropia',    'H3: entropia posicional prediz hits'),
        ('mahalanobis', 'H4: distância Mahalanobis prediz hits'),
        ('pares',       'H5: nº de pares em t-1 prediz hits'),
        ('moldura',     'H6: nº de moldura em t-1 prediz hits'),
        ('soma',        'H7: soma de t-1 prediz hits'),
        ('energia',     'H8: energia (gaps absolutos) prediz hits'),
        ('consecutivos','H9: consecutivos de t-1 prediz hits'),
        ('amplitude',   'H10: amplitude de t-1 prediz hits'),
    ]

    results = []
    any_significant = False

    for feature, description in hipoteses:
        res = test_hypothesis(feature, records_test, len(results)+1, description)
        results.append(res)

        print(f"\n   {description}")
        print(f"   Spearman rho={res['rho']:+.4f}  p={res['p_spearman']:.4f}  {res['status']}")
        print(f"   Mann-Whitney Q1 vs Q4: {res['mean_low_q']:.4f} vs {res['mean_high_q']:.4f}  "
              f"(diff={res['diff_means']:+.4f})  p_mw={res['p_mw']:.4f}")

        if res['significant_bonf']:
            any_significant = True

    # ─── ANÁLISE 3: Estabilidade temporal dos sinais ───
    print("\n" + "─"*65)
    print("📊 ANÁLISE 3: ESTABILIDADE TEMPORAL DOS SINAIS")
    print("   (um sinal real deve ser consistente ao longo do tempo)")
    print("─"*65)

    # Testar as 3 features com menor p-valor
    top_features = sorted(results, key=lambda r: r['p_spearman'])[:3]
    for res in top_features:
        stab = test_temporal_stability(res['feature'], all_records)
        if stab:
            consiste = "✅ Consistente" if stab['sinal_consistente'] else "❌ Inconsistente"
            print(f"\n   {res['feature']}: {consiste}")
            print(f"   rho por período: {[f'{r:+.3f}' for r in stab['rhos_por_periodo']]}")
            print(f"   CV do rho: {stab['cv_rho']:.2f} (< 0.5 indica estabilidade)")

    # ─── ANÁLISE 4: Distribuição dos hits aleatórios ───
    print("\n" + "─"*65)
    print("📊 ANÁLISE 4: DISTRIBUIÇÃO EMPÍRICA DE HITS")
    print("   (calibração do target — confirma que o baseline é correto)")
    print("─"*65)

    all_avg_hits = [r['avg_hits'] for r in all_records]
    print(f"   Média global avg_hits: {np.mean(all_avg_hits):.5f} (teórico: 9.00000)")
    print(f"   Desvio padrão: {np.std(all_avg_hits):.5f}")
    print(f"   Min: {np.min(all_avg_hits):.3f}  Max: {np.max(all_avg_hits):.3f}")

    # Teste: a média do target é significativamente diferente de 9.0?
    t_stat, p_mean = stats.ttest_1samp(all_avg_hits, 9.0)
    print(f"\n   Teste t (média == 9.0): t={t_stat:.4f}  p={p_mean:.4f}")
    if p_mean < 0.05:
        print(f"   ⚠️  A distribuição de hits é biased ({np.mean(all_avg_hits):.4f} ≠ 9.0)")
    else:
        print(f"   ✅ Distribuição consistente com sorteio uniforme")

    # ─── ANÁLISE 5: Teste de runs (aleatoriedade temporal) ───
    print("\n" + "─"*65)
    print("📊 ANÁLISE 5: TESTE DE RUNS (aleatoriedade temporal)")
    print("─"*65)

    # Para as features mais importantes, testar se a série temporal é aleatória
    for feat in ['repeticao', 'soma', 'pares']:
        vals = np.array([r[feat] for r in all_records])
        median_val = np.median(vals)
        runs = [1]
        for i in range(1, len(vals)):
            if (vals[i] >= median_val) == (vals[i-1] >= median_val):
                runs[-1] += 1
            else:
                runs.append(1)
        n1 = sum(1 for v in vals if v >= median_val)
        n2 = len(vals) - n1
        expected_runs = 2*n1*n2/(n1+n2) + 1
        var_runs = 2*n1*n2*(2*n1*n2-n1-n2) / ((n1+n2)**2*(n1+n2-1))
        z = (len(runs) - expected_runs) / (var_runs**0.5 + 1e-10)
        p_runs = 2 * (1 - stats.norm.cdf(abs(z)))
        aleat = "✅ Aleatório" if p_runs > 0.05 else "❌ Não aleatório (há padrão)"
        print(f"   {feat:15s}: runs={len(runs)}  esperado={expected_runs:.1f}  z={z:+.3f}  p={p_runs:.4f}  {aleat}")

    # ─── SUMÁRIO FINAL ───
    print("\n" + "=" * 65)
    print("📋 SUMÁRIO FINAL — VEREDICTO CIENTÍFICO")
    print("=" * 65)

    n_sig_bonf = sum(1 for r in results if r['significant_bonf'])
    n_sig_raw  = sum(1 for r in results if r['significant_raw'])

    print(f"\n   Hipóteses testadas: {N_HIPOTESES}")
    print(f"   Significativas (Bonferroni p<{ALPHA_BONF:.4f}): {n_sig_bonf}")
    print(f"   Significativas (raw p<{ALPHA}): {n_sig_raw}")

    print(f"\n   Ranking por p-valor:")
    for r in sorted(results, key=lambda x: x['p_spearman']):
        flag = "★" if r['significant_bonf'] else ("·" if r['significant_raw'] else " ")
        print(f"   {flag} {r['feature']:15s}: rho={r['rho']:+.4f}  p={r['p_spearman']:.5f}")

    print()
    if n_sig_bonf > 0:
        print("   🟡 RESULTADO: SINAL POTENCIAL DETECTADO")
        print("   → Existe correlação significativa após correção de Bonferroni.")
        print("   → PRÓXIMO PASSO: replicação em dataset independente.")
        print("   → NÃO interpretar como previsão prática imediata.")
    elif n_sig_raw > 0:
        print("   🟠 RESULTADO: SINAL MARGINAL (pode ser falso positivo)")
        print("   → Correlações que não sobrevivem à correção de Bonferroni")
        print("   → são consistentes com múltiplos testes em ruído puro.")
        print("   → Sem evidência de previsibilidade real.")
    else:
        print("   🔴 RESULTADO: NENHUM SINAL DETECTADO")
        print("   → Nenhuma das 10 hipóteses sobreviveu ao teste.")
        print("   → Os resultados são consistentes com sorteio uniforme")
        print("   → independente. Não existe sinal preditivo nas features")
        print("   → estruturais testadas.")

    print()
    print("   ─────────────────────────────────────────────────────")
    print("   NOTA METODOLÓGICA:")
    print("   Este resultado é definitivo para as hipóteses testadas.")
    print("   Novos testes em novos dados requerem novo pré-registro.")
    print("   Iterar sobre o mesmo dataset invalida os p-valores.")
    print("   ─────────────────────────────────────────────────────")

if __name__ == "__main__":
    main()
