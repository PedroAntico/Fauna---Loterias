#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRONT-END INTERATIVO - LOTOFÁCIL
=================================
Versão 1.0 - Interface Estratégica para o Motor de Otimização

Este módulo importa o motor e fornece interface interativa
para filtragem, ranking e exibição de jogos.
"""

from lotofacil_core import (
    LotofacilOptimizerV3,
    contar_pares,
    contar_impares,
    contar_primos,
    contar_moldura,
    contar_centro,
    contar_repetidos,
    calcular_soma,
    calcular_amplitude,
    contar_consecutivos,
    TOTAL_NUMBERS,
    NUMBERS_PER_GAME,
    PRIMES,
    MOLDURA,
    CENTRO,
    QUADRANTES
)
import numpy as np
from datetime import datetime
import os


class LotofacilFrontend:
    """
    Front-end interativo para o motor de otimização
    
    Funcionalidades:
    - Coleta de preferências do usuário
    - Filtragem de candidatos
    - Ranking por fitness
    - Exibição formatada
    - Exportação de resultados
    """
    
    def __init__(self):
        """Inicializa o front-end com o motor de otimização"""
        print("\n" + "="*70)
        print("🎯 GERADOR ESTRATÉGICO - LOTOFÁCIL")
        print("   Front-end Interativo v1.0")
        print("="*70)
        
        # Inicializar motor
        print("\n📂 Inicializando motor de otimização...")
        self.optimizer = LotofacilOptimizerV3()
        
        # Dados do último concurso
        self.ultimo_concurso = self.optimizer.get_last_draw()
        self.ultimos_concursos = self.optimizer.get_last_draws(5)
        
        if self.ultimo_concurso:
            print(f"📅 Último concurso carregado: {sorted(self.ultimo_concurso)}")
        
        # Preferências do usuário
        self.preferences = {}
        
        # Resultados
        self.candidates = []
        self.filtered = []
        self.ranked = []
    
    def show_historical_info(self):
        """Mostra informações históricas relevantes"""
        print("\n" + "-"*50)
        print("📊 INFORMAÇÕES HISTÓRICAS")
        print("-"*50)
        
        # Frequência
        freq = self.optimizer.get_historical_frequency()
        if freq:
            top5 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
            bottom5 = sorted(freq.items(), key=lambda x: x[1])[:5]
            
            print(f"\n🔥 Mais frequentes: {[x[0] for x in top5]}")
            print(f"❄️  Menos frequentes: {[x[0] for x in bottom5]}")
        
        # Últimos concursos
        if self.ultimos_concursos:
            print(f"\n📅 Últimos 5 concursos:")
            for i, draw in enumerate(self.ultimos_concursos, 1):
                print(f"   {i}. {sorted(draw)}")
    
    def collect_preferences(self):
        """
        Coleta preferências do usuário de forma interativa
        
        Opções:
        - Faixas de pares/ímpares
        - Faixas de primos
        - Moldura/centro
        - Repetidos do último concurso
        - Dezenas fixas
        - Dezenas excluídas
        - Número de jogos a gerar
        """
        print("\n" + "="*50)
        print("🎯 CONFIGURAÇÃO DE PREFERÊNCIAS")
        print("="*50)
        print("\n💡 Dica: Pressione ENTER para usar valores padrão (sem restrição)")
        
        # 1. Pares
        print("\n📊 PARES/ÍMPARES")
        print("   (Distribuição típica: 6-9 pares)")
        pares_min = self._input_int("   Pares mínimos [6]: ", 0, 15, 6)
        pares_max = self._input_int("   Pares máximos [9]: ", pares_min, 15, 9)
        self.preferences['pares_range'] = (pares_min, pares_max)
        
        # 2. Primos
        print("\n🔢 PRIMOS")
        print(f"   (Primos disponíveis: {sorted(PRIMES)})")
        print("   (Distribuição típica: 3-6 primos)")
        primos_min = self._input_int("   Primos mínimos [3]: ", 0, 15, 3)
        primos_max = self._input_int("   Primos máximos [6]: ", primos_min, 15, 6)
        self.preferences['primos_range'] = (primos_min, primos_max)
        
        # 3. Moldura/Centro
        print("\n🖼️  MOLDURA/CENTRO")
        print(f"   Moldura: {sorted(MOLDURA)}")
        print(f"   Centro: {sorted(CENTRO)}")
        print("   (Distribuição típica: 7-10 moldura)")
        moldura_min = self._input_int("   Moldura mínima [7]: ", 0, 15, 7)
        moldura_max = self._input_int("   Moldura máxima [10]: ", moldura_min, 15, 10)
        self.preferences['moldura_range'] = (moldura_min, moldura_max)
        
        # 4. Repetidos
        if self.ultimo_concurso:
            print(f"\n🔄 REPETIDOS DO ÚLTIMO CONCURSO")
            print(f"   Último: {sorted(self.ultimo_concurso)}")
            print("   (Distribuição típica: 8-10 repetidos)")
            rep_min = self._input_int("   Repetidos mínimos [8]: ", 0, 15, 8)
            rep_max = self._input_int("   Repetidos máximos [10]: ", rep_min, 15, 10)
            self.preferences['repetidos_range'] = (rep_min, rep_max)
        
        # 5. Dezenas fixas
        print("\n📌 DEZENAS FIXAS")
        print("   Digite as dezenas separadas por espaço (ou ENTER para nenhuma)")
        fixas_input = input("   Dezenas fixas: ").strip()
        
        if fixas_input:
            try:
                fixas = sorted(set(int(x) for x in fixas_input.split() if 1 <= int(x) <= 25))
                self.preferences['fixas'] = fixas[:15]  # Máximo 15
                print(f"   ✅ Fixas: {fixas}")
            except ValueError:
                print("   ⚠️  Entrada inválida. Nenhuma dezena fixa.")
                self.preferences['fixas'] = []
        else:
            self.preferences['fixas'] = []
        
        # 6. Dezenas excluídas
        print("\n🚫 DEZENAS EXCLUÍDAS")
        print("   Digite as dezenas a EXCLUIR (ou ENTER para nenhuma)")
        excluir_input = input("   Dezenas excluídas: ").strip()
        
        if excluir_input:
            try:
                excluidas = set(int(x) for x in excluir_input.split() if 1 <= int(x) <= 25)
                # Não pode excluir as fixas
                excluidas -= set(self.preferences.get('fixas', []))
                self.preferences['excluidas'] = excluidas
                print(f"   ✅ Excluídas: {sorted(excluidas)}")
            except ValueError:
                print("   ⚠️  Entrada inválida. Nenhuma dezena excluída.")
                self.preferences['excluidas'] = set()
        else:
            self.preferences['excluidas'] = set()
        
        # 7. Número de jogos
        print("\n🎯 GERAÇÃO")
        n_jogos = self._input_int("   Número de jogos a gerar [10]: ", 1, 100, 10)
        self.preferences['n_jogos'] = n_jogos
        
        print("\n✅ Preferências configuradas!")
    
    def _input_int(self, prompt, min_val, max_val, default):
        """
        Input seguro de inteiro
        
        Args:
            prompt: Mensagem
            min_val: Valor mínimo
            max_val: Valor máximo
            default: Valor padrão
            
        Returns:
            int: Valor inserido
        """
        while True:
            user_input = input(prompt).strip()
            
            if not user_input:
                return default
            
            try:
                value = int(user_input)
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"   ⚠️  Valor deve estar entre {min_val} e {max_val}")
            except ValueError:
                print("   ⚠️  Digite um número inteiro válido")
    
    def generate_and_filter(self):
        """
        Gera candidatos e aplica filtros
        
        Pipeline:
        1. Gera milhares de candidatos
        2. Filtra por preferências
        3. Armazena filtrados
        """
        print("\n" + "="*50)
        print("🔍 GERANDO E FILTRANDO CANDIDATOS...")
        print("="*50)
        
        # Gerar candidatos
        n_generate = 10000
        print(f"\n🎲 Gerando {n_generate} candidatos...")
        self.candidates = self.optimizer.generate_candidates(n_generate)
        print(f"   ✅ {len(self.candidates)} candidatos gerados")
        
        # Aplicar filtros
        print("\n🔍 Aplicando filtros...")
        self.filtered = []
        
        pref = self.preferences
        
        for game in self.candidates:
            # Verificar dezenas fixas
            if pref.get('fixas'):
                if not all(d in game for d in pref['fixas']):
                    continue
            
            # Verificar dezenas excluídas
            if pref.get('excluidas'):
                if any(d in game for d in pref['excluidas']):
                    continue
            
            # Verificar pares
            if 'pares_range' in pref:
                pares = contar_pares(game)
                if not (pref['pares_range'][0] <= pares <= pref['pares_range'][1]):
                    continue
            
            # Verificar primos
            if 'primos_range' in pref:
                primos = contar_primos(game)
                if not (pref['primos_range'][0] <= primos <= pref['primos_range'][1]):
                    continue
            
            # Verificar moldura
            if 'moldura_range' in pref:
                moldura = contar_moldura(game)
                if not (pref['moldura_range'][0] <= moldura <= pref['moldura_range'][1]):
                    continue
            
            # Verificar repetidos
            if 'repetidos_range' in pref and self.ultimo_concurso:
                rep = contar_repetidos(game, self.ultimo_concurso)
                if not (pref['repetidos_range'][0] <= rep <= pref['repetidos_range'][1]):
                    continue
            
            # Passou por todos os filtros
            self.filtered.append(game)
        
        print(f"   ✅ {len(self.filtered)} jogos após filtros")
        
        if len(self.filtered) == 0:
            print("\n⚠️  NENHUM jogo passou nos filtros!")
            print("   Tente relaxar as restrições.")
            return False
        
        return True
    
    def rank_and_display(self):
        """
        Rankeia jogos filtrados e exibe os melhores
        """
        if not self.filtered:
            print("\n⚠️  Nenhum jogo para rankear.")
            return
        
        print("\n" + "="*50)
        print("🏆 RANKEANDO JOGOS...")
        print("="*50)
        
        # Rankear
        self.ranked = self.optimizer.rank_games(
            self.filtered, 
            top_n=self.preferences.get('n_jogos', 10)
        )
        
        # Exibir
        self._display_ranked()
    
    def _display_ranked(self):
        """Exibe os jogos rankeados formatados"""
        if not self.ranked:
            return
        
        n = len(self.ranked)
        
        print(f"\n{'='*70}")
        print(f"🎯 TOP {n} JOGOS RECOMENDADOS")
        print(f"{'='*70}")
        
        for i, (fitness, game, metrics) in enumerate(self.ranked, 1):
            sorted_game = sorted(game)
            
            pares = contar_pares(game)
            impares = contar_impares(game)
            primos = contar_primos(game)
            moldura = contar_moldura(game)
            centro = contar_centro(game)
            soma = calcular_soma(game)
            amplitude = calcular_amplitude(game)
            consecutivos = contar_consecutivos(game)
            
            if self.ultimo_concurso:
                repetidos = contar_repetidos(game, self.ultimo_concurso)
            else:
                repetidos = 0
            
            print(f"\n{'─'*50}")
            print(f"JOGO {i:02d} | Fitness: {fitness:.2f}")
            print(f"{'─'*50}")
            print(f"  Dezenas: {sorted_game}")
            print(f"  ─────────────────────────────────")
            print(f"  Pares: {pares} | Ímpares: {impares} | Primos: {primos}")
            print(f"  Moldura: {moldura} | Centro: {centro}")
            print(f"  Soma: {soma} | Amplitude: {amplitude}")
            print(f"  Consecutivos: {consecutivos} | Repetidos: {repetidos}")
            print(f"  Penalidade estrutural: {metrics['penalty']:.1f}")
        
        # Resumo estatístico
        print(f"\n{'='*70}")
        print(f"📊 RESUMO ESTATÍSTICO DOS {n} JOGOS")
        print(f"{'='*70}")
        
        all_pares = [contar_pares(g) for _, g, _ in self.ranked]
        all_primos = [contar_primos(g) for _, g, _ in self.ranked]
        all_moldura = [contar_moldura(g) for _, g, _ in self.ranked]
        all_somas = [calcular_soma(g) for _, g, _ in self.ranked]
        
        print(f"  Pares: {min(all_pares)}-{max(all_pares)} (média: {np.mean(all_pares):.1f})")
        print(f"  Primos: {min(all_primos)}-{max(all_primos)} (média: {np.mean(all_primos):.1f})")
        print(f"  Moldura: {min(all_moldura)}-{max(all_moldura)} (média: {np.mean(all_moldura):.1f})")
        print(f"  Soma: {min(all_somas)}-{max(all_somas)} (média: {np.mean(all_somas):.1f})")
    
    def export_results(self):
        """Exporta resultados para arquivo"""
        if not self.ranked:
            print("\n⚠️  Nenhum resultado para exportar.")
            return
        
        print("\n💾 Exportando resultados...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'jogos_recomendados_{timestamp}.txt'
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("JOGOS RECOMENDADOS - LOTOFÁCIL\n")
            f.write(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
            f.write("="*50 + "\n\n")
            
            if self.ultimo_concurso:
                f.write(f"Último concurso: {sorted(self.ultimo_concurso)}\n\n")
            
            f.write("PREFERÊNCIAS:\n")
            for key, value in self.preferences.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            for i, (fitness, game, metrics) in enumerate(self.ranked, 1):
                sorted_game = sorted(game)
                f.write(f"Jogo {i:02d}: {sorted_game}\n")
                f.write(f"  Fitness: {fitness:.2f}\n")
                f.write(f"  Pares: {contar_pares(game)}\n")
                f.write(f"  Primos: {contar_primos(game)}\n")
                f.write(f"  Soma: {calcular_soma(game)}\n")
                f.write("\n")
        
        print(f"✅ Resultados exportados para: {filename}")
    
    def run(self):
        """
        Executa o fluxo completo do front-end
        
        Pipeline:
        1. Mostrar informações históricas
        2. Coletar preferências
        3. Gerar e filtrar candidatos
        4. Rankear e exibir
        5. Exportar resultados
        """
        # 1. Informações históricas
        self.show_historical_info()
        
        # 2. Coletar preferências
        self.collect_preferences()
        
        # 3. Gerar e filtrar
        success = self.generate_and_filter()
        
        if not success:
            print("\n❌ Processo interrompido. Ajuste os filtros e tente novamente.")
            return
        
        # 4. Rankear e exibir
        self.rank_and_display()
        
        # 5. Exportar
        export = input("\n💾 Deseja exportar os resultados? (S/n): ").strip().lower()
        if export != 'n':
            self.export_results()
        
        print("\n" + "="*70)
        print("✅ PROCESSO CONCLUÍDO!")
        print("="*70)
        print("\n💡 LEMBRE-SE:")
        print("   • Este sistema NÃO prevê resultados futuros")
        print("   • Otimiza cobertura e diversidade estrutural")
        print("   • Jogue com responsabilidade!")
        print("="*70)


def main():
    """Função principal"""
    frontend = LotofacilFrontend()
    frontend.run()


if __name__ == "__main__":
    main()
