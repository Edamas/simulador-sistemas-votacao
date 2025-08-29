import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from collections import Counter
import time
import string
from itertools import combinations

# --- Configurações da Página ---
st.set_page_config(
    page_title="Simulador de Sistemas de Votação",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Funções de Lógica Generalizadas ---

def generate_agents(num_agents, num_candidates):
    """Gera um DataFrame de agentes com preferências aleatórias para N candidatos."""
    candidates = list(string.ascii_uppercase[:num_candidates])
    data = np.random.uniform(0, 1, (num_agents, num_candidates))
    df = pd.DataFrame(data, columns=[f"pref_{c}" for c in candidates])
    
    def get_ranking(row):
        prefs = {candidate: row[f"pref_{candidate}"] for candidate in candidates}
        return sorted(prefs, key=prefs.get, reverse=True)

    rankings = df.apply(get_ranking, axis=1, result_type='expand')
    rankings.columns = [f'rank_{i+1}' for i in range(num_candidates)]
    df = pd.concat([df, rankings], axis=1)
    df['Convicção (Desvio Padrão das Prefs.)'] = df[[f"pref_{c}" for c in candidates]].std(axis=1)
    return df, candidates

def break_tie(tied_candidates, tie_breaker_method):
    """Resolve empates com base no método escolhido."""
    if not tied_candidates: return "N/A"
    if tie_breaker_method == "Aleatório":
        return np.random.choice(tied_candidates)
    elif tie_breaker_method == "Ordem Alfabética":
        return sorted(tied_candidates)[0]
    else: # Anulação da Votação
        return "Anulada"

# --- LÓGICA DE VOTO ESTRATÉGICO GENERALIZADA ---
def apply_strategic_vote(df, candidates, strategic_prob):
    """Aplica uma estratégia de 'Compromisso' onde eleitores de candidatos fracos promovem sua 2ª opção."""
    if strategic_prob == 0: return df

    strategic_df = df.copy()
    
    honest_counts = df['rank_1'].value_counts(normalize=True)
    weak_threshold = honest_counts.quantile(0.33)
    weak_candidates = honest_counts[honest_counts <= weak_threshold].index.tolist()

    strategic_voter_indices = df[df['rank_1'].isin(weak_candidates)].index

    for idx in strategic_voter_indices:
        if np.random.rand() < strategic_prob:
            original_rank_1 = strategic_df.loc[idx, 'rank_1']
            original_rank_2 = strategic_df.loc[idx, 'rank_2']
            strategic_df.loc[idx, 'rank_1'] = original_rank_2
            strategic_df.loc[idx, 'rank_2'] = original_rank_1
            
    return strategic_df

# --- Funções dos Métodos de Votação (simplificadas para análise estatística) ---
def run_plurality(df, candidates, tie_breaker_method):
    counts = df['rank_1'].value_counts()
    max_votes = counts.max()
    tied = counts[counts == max_votes].index.tolist()
    winner = break_tie(tied, tie_breaker_method) if len(tied) > 1 else tied[0]
    return winner, counts, len(tied) > 1, df['rank_1']

def run_two_round_runoff(df, candidates, tie_breaker_method):
    num_voters = len(df)
    counts_r1 = df['rank_1'].value_counts()
    if counts_r1.empty:
        return "N/A", {}, False, pd.Series()
    if counts_r1.max() > num_voters / 2:
        max_votes = counts_r1.max()
        tied = counts_r1[counts_r1 == max_votes].index.tolist()
        winner = break_tie(tied, tie_breaker_method) if len(tied) > 1 else tied[0]
        return winner, {"Turno 1": counts_r1}, len(tied) > 1, df['rank_1']
    top_2 = counts_r1.index[:2].tolist()
    if len(top_2) < 2: # Menos de 2 candidatos, o primeiro vence
        return counts_r1.index[0], {"Turno 1": counts_r1}, False, df['rank_1']
    round_2_votes = df.apply(lambda row: top_2[0] if row[f'pref_{top_2[0]}'] > row[f'pref_{top_2[1]}'] else top_2[1], axis=1)
    counts_r2 = round_2_votes.value_counts()
    max_votes_r2 = counts_r2.max()
    tied_r2 = counts_r2[counts_r2 == max_votes_r2].index.tolist()
    winner = break_tie(tied_r2, tie_breaker_method) if len(tied_r2) > 1 else tied_r2[0]
    results = {"Turno 1": counts_r1, "Finalistas": top_2, "Turno 2": counts_r2}
    return winner, results, len(tied_r2) > 1, round_2_votes

def run_irv(df, candidates, tie_breaker_method):
    num_voters = len(df)
    rank_cols = [f'rank_{i+1}' for i in range(len(candidates))]
    votes_df = df[rank_cols].copy() # MUDANÇA: Copia o DataFrame
    eliminated_candidates = []
    rounds_summary = []
    for i in range(len(candidates) - 1):
        counts = votes_df['rank_1'].value_counts()
        if counts.empty: break
        rounds_summary.append(counts)
        if counts.max() > num_voters / 2:
            max_votes = counts.max()
            tied = counts[counts == max_votes].index.tolist()
            winner = break_tie(tied, tie_breaker_method) if len(tied) > 1 else tied[0]
            return winner, rounds_summary, len(tied) > 1, votes_df['rank_1']
        if len(counts) <= 2: break
        last_place_votes = counts.min()
        tied_for_last = counts[counts == last_place_votes].index.tolist()
        eliminated = break_tie(tied_for_last, "Aleatório")
        eliminated_candidates.append(eliminated)
        for idx, row in votes_df.iterrows(): # MUDANÇA: Itera sobre o DataFrame
            if row['rank_1'] == eliminated:
                for rank_idx in range(2, len(candidates) + 1):
                    next_best = row.get(f'rank_{rank_idx}')
                    if next_best and next_best not in eliminated_candidates:
                        votes_df.loc[idx, 'rank_1'] = next_best
                        break
    final_counts = votes_df['rank_1'].value_counts()
    if final_counts.empty: return "N/A", rounds_summary, False, pd.Series()
    max_final_votes = final_counts.max()
    tied_final = final_counts[final_counts == max_final_votes].index.tolist()
    winner = break_tie(tied_final, tie_breaker_method) if len(tied_final) > 1 else tied_final[0]
    if not rounds_summary or not rounds_summary[-1].equals(final_counts):
        rounds_summary.append(final_counts)
    return winner, rounds_summary, len(tied_final) > 1, votes_df['rank_1']

def run_approval(df, candidates, tie_breaker_method):
    votes = df.apply(lambda row: tuple(sorted([c for c in candidates if row[f'pref_{c}'] > 0.5])), axis=1)
    all_approvals = Counter(c for vote_tuple in votes for c in vote_tuple)
    if not all_approvals: return "Ninguém", pd.Series(), False, pd.Series([()]*len(df))
    counts = pd.Series(all_approvals).sort_values(ascending=False)
    max_approvals = counts.max()
    tied = counts[counts == max_approvals].index.tolist()
    winner = break_tie(tied, tie_breaker_method) if len(tied) > 1 else tied[0]
    return winner, counts, len(tied) > 1, votes

def run_score(df, candidates, tie_breaker_method):
    scores = pd.DataFrame()
    for c in candidates:
        scores[c] = df[f'pref_{c}'] * 10
    total_scores = scores.sum().sort_values(ascending=False)
    max_score = total_scores.max()
    tied = total_scores[total_scores == max_score].index.tolist()
    winner = break_tie(tied, tie_breaker_method) if len(tied) > 1 else tied[0]
    return winner, total_scores, len(tied) > 1, df['rank_1']

def run_borda(df, candidates, tie_breaker_method):
    points = {c: 0 for c in candidates}
    num_candidates = len(candidates)
    for i in range(num_candidates):
        score = num_candidates - 1 - i
        counts = df[f'rank_{i+1}'].value_counts()
        for candidate, num_votes in counts.items():
            points[candidate] += num_votes * score
    counts = pd.Series(points).sort_values(ascending=False)
    max_points = counts.max()
    tied = counts[counts == max_points].index.tolist()
    winner = break_tie(tied, tie_breaker_method) if len(tied) > 1 else tied[0]
    return winner, counts, len(tied) > 1, df['rank_1']

def run_condorcet_base(df, candidates):
    pairs = list(combinations(candidates, 2))
    wins = {c: 0 for c in candidates}
    for c1, c2 in pairs:
        c1_votes = (df[f'pref_{c1}'] > df[f'pref_{c2}']).sum()
        c2_votes = (df[f'pref_{c2}'] > df[f'pref_{c1}']).sum()
        if c1_votes > c2_votes: wins[c1] += 1
        elif c2_votes > c1_votes: wins[c2] += 1
    return wins

def run_condorcet(df, candidates, tie_breaker_method):
    wins = run_condorcet_base(df, candidates)
    for candidate, num_wins in wins.items():
        if num_wins == len(candidates) - 1:
            return candidate, pd.Series(wins), False, df['rank_1']
    winner = break_tie([], tie_breaker_method) # Paradox is a failure, not a tie to break
    return winner, pd.Series(wins), True, df['rank_1']

def run_copeland(df, candidates, tie_breaker_method):
    wins = run_condorcet_base(df, candidates)
    counts = pd.Series(wins).sort_values(ascending=False)
    max_wins = counts.max()
    tied = counts[counts == max_wins].index.tolist()
    winner = break_tie(tied, tie_breaker_method) if len(tied) > 1 else tied[0]
    return winner, counts, len(tied) > 1, df['rank_1']

def run_contingent(df, candidates, tie_breaker_method):
    num_voters = len(df)
    counts_r1 = df['rank_1'].value_counts()
    if counts_r1.empty: return "N/A", {}, False, pd.Series()
    if counts_r1.max() > num_voters / 2:
        max_votes = counts_r1.max()
        tied = counts_r1[counts_r1 == max_votes].index.tolist()
        winner = break_tie(tied, tie_breaker_method) if len(tied) > 1 else tied[0]
        return winner, {"Turno 1": counts_r1}, len(tied) > 1, df['rank_1']
    top_2 = counts_r1.index[:2].tolist()
    if len(top_2) < 2: return counts_r1.index[0], {"Turno 1": counts_r1}, False, df['rank_1']
    contingent_votes_series = df['rank_1'].copy() # MUDANÇA: Usa Series
    eliminated_voters = contingent_votes_series[~contingent_votes_series.isin(top_2)].index
    for idx in eliminated_voters:
        # MUDANÇA: Acessa diretamente a Series
        if df.loc[idx, f'pref_{top_2[0]}'] > df.loc[idx, f'pref_{top_2[1]}']:
            contingent_votes_series.loc[idx] = top_2[0]
        else:
            contingent_votes_series.loc[idx] = top_2[1]
    final_counts = contingent_votes_series.value_counts()
    if final_counts.empty: # MUDANÇA: Trata caso de final_counts vazio
        return "Anulada", {"Turno 1": counts_r1, "Resultado Final": final_counts}, True, contingent_votes_series
    max_final_votes = final_counts.max()
    tied_final = final_counts[final_counts == max_final_votes].index.tolist()
    winner = break_tie(tied_final, tie_breaker_method) if len(tied_final) > 1 else tied_final[0]
    return winner, {"Turno 1": counts_r1, "Resultado Final": final_counts}, len(tied_final) > 1, contingent_votes_series

def run_anti_plurality(df, candidates, tie_breaker_method):
    last_rank_col = f'rank_{len(candidates)}'
    against_votes = df[last_rank_col].value_counts().sort_values(ascending=True)
    min_votes = against_votes.min()
    tied = against_votes[against_votes == min_votes].index.tolist()
    winner = break_tie(tied, tie_breaker_method) if len(tied) > 1 else tied[0]
    return winner, against_votes, len(tied) > 1, df[last_rank_col]

# --- Funções de Visualização ---
def create_2d_plot(df, candidates, title, declared_votes=None):
    c1, c2 = candidates
    fig = go.Figure()
    plot_df = df
    if declared_votes is not None:
        mapping = {c1: [1, 0], c2: [0, 1]}
        positions = declared_votes.map(mapping).tolist()
        declared_df = pd.DataFrame(positions, columns=[f'pref_{c1}', f'pref_{c2}'])
        plot_df = declared_df
        for i in range(len(df)):
            fig.add_trace(go.Scatter(x=[df[f'pref_{c1}'].iloc[i], declared_df[f'pref_{c1}'].iloc[i]], y=[df[f'pref_{c2}'].iloc[i], declared_df[f'pref_{c2}'].iloc[i]], mode='lines', line=dict(width=1, color='rgba(128,128,128,0.5)'), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=plot_df[f'pref_{c1}'], y=plot_df[f'pref_{c2}'], mode='markers', marker=dict(size=2, color=df['Convicção (Desvio Padrão das Prefs.)'], colorscale='Plasma', showscale=True, colorbar=dict(title='Convicção')), hoverinfo='none'))
    fig.update_layout(title=title, xaxis_title=f'Preferência {c1}', yaxis_title=f'Preferência {c2}', xaxis=dict(range=[-0.1, 1.1]), yaxis=dict(range=[-0.1, 1.1]))
    return fig

def create_3d_plot(df, candidates, title, declared_votes=None):
    c1, c2, c3 = candidates
    fig = go.Figure()
    plot_df = df
    if declared_votes is not None:
        mapping = {c1: [1,0,0], c2: [0,1,0], c3: [0,0,1]}
        positions = declared_votes.map(mapping).tolist()
        declared_df = pd.DataFrame(positions, columns=[f'pref_{c1}', f'pref_{c2}', f'pref_{c3}'])
        plot_df = declared_df
        for i in range(len(df)):
            fig.add_trace(go.Scatter3d(x=[df[f'pref_{c1}'].iloc[i], declared_df[f'pref_{c1}'].iloc[i]], y=[df[f'pref_{c2}'].iloc[i], declared_df[f'pref_{c2}'].iloc[i]], z=[df[f'pref_{c3}'].iloc[i], declared_df[f'pref_{c3}'].iloc[i]], mode='lines', line=dict(width=1, color='rgba(128,128,128,0.5)'), hoverinfo='none'))
    fig.add_trace(go.Scatter3d(x=plot_df[f'pref_{c1}'], y=plot_df[f'pref_{c2}'], z=plot_df[f'pref_{c3}'], mode='markers', marker=dict(size=2, color=df['Convicção (Desvio Padrão das Prefs.)'], colorscale='Plasma', showscale=True, colorbar=dict(title='Convicção')), hoverinfo='none'))
    fig.update_layout(title=title, scene=dict(xaxis_title=f'Pref {c1}', yaxis_title=f'Pref {c2}', zaxis_title=f'Pref {c3}', aspectmode='cube', xaxis=dict(range=[-0.1, 1.1]), yaxis=dict(range=[-0.1, 1.1]), zaxis=dict(range=[-0.1, 1.1])))
    return fig

def create_parallel_coordinates_plot(df, candidates, title, declared_votes=None):
    fig = go.Figure()
    dims = [dict(label=c, values=df[f'pref_{c}'], range=[0,1]) for c in candidates]
    color_values, colorscale = df['Convicção (Desvio Padrão das Prefs.)'], 'Plasma'
    if declared_votes is not None:
        candidate_map = {name: i for i, name in enumerate(candidates)}
        color_values = declared_votes.map(candidate_map)
        colorscale = 'Rainbow'
    fig.add_trace(go.Parcoords(line=dict(color=color_values, colorscale=colorscale, showscale=True, colorbar=dict(title='Convicção' if declared_votes is None else 'Voto')), dimensions=dims))
    fig.update_layout(title=title)
    return fig

# --- Interface Principal ---
st.title("📊 Ferramenta de Análise Estatística de Sistemas de Votação")

methods_to_analyze = {
    "Pluralidade": run_plurality, "Dois Turnos (Runoff)": run_two_round_runoff, "Voto Ranqueado (IRV)": run_irv,
    "Voto por Aprovação": run_approval, "Voto por Pontuação": run_score, "Contagem de Borda": run_borda,
    "Voto Contingente": run_contingent, "Condorcet": run_condorcet, "Método de Copeland": run_copeland,
    "Anti-Pluralidade": run_anti_plurality
}

with st.sidebar:
    st.header("Configurações Gerais")
    num_agents = st.number_input("Número de Agentes por Simulação", min_value=2, max_value=10000, value=100, step=1)
    num_candidates = st.slider("Número de Candidatos", 3, 10, 3, 1)
    strategic_prob = st.slider("Probabilidade de Voto Estratégico", 0.0, 1.0, 0.25, 0.05, help="Simula o 'voto útil' de compromisso.")
    tie_breaker_method = st.selectbox("Em caso de Empate/Paradoxo", ["Anulação da Votação", "Aleatório", "Ordem Alfabética"])
    
    st.markdown("---")
    st.header("Análise Estatística")
    num_simulations = st.slider("Número de Simulações", 10, 10000, 100, 10)
    run_analysis = st.button("Executar Análise Estatística", type="primary")

# --- Lógica Principal ---

if 'last_analysis_summary' not in st.session_state:
    st.session_state.last_analysis_summary = None
    st.session_state.last_example_df = None
    st.session_state.last_example_candidates = None

if run_analysis:
    st.header("Resultados da Análise Estatística")
    progress_bar = st.progress(0, text="Iniciando simulações...")
    
    results_accumulator = []
    last_run_df, last_run_candidates = None, None

    for i in range(num_simulations):
        agents_df, candidates = generate_agents(num_agents, num_candidates)
        strategic_df = apply_strategic_vote(agents_df, candidates, strategic_prob)
        
        condorcet_winner = run_condorcet(agents_df, candidates, tie_breaker_method)[0]
        if condorcet_winner == "Paradoxo": condorcet_winner = None
        
        utilitarian_winner = run_score(agents_df, candidates, tie_breaker_method)[0]
        if utilitarian_winner == "Anulada": utilitarian_winner = None

        majority_counts = agents_df['rank_1'].value_counts()
        majority_winner = majority_counts.index[0] if not majority_counts.empty and majority_counts.iloc[0] > num_agents / 2 else None

        plurality_winner = run_plurality(agents_df, candidates, tie_breaker_method)[0]
        if plurality_winner == "Anulada": plurality_winner = None

        for name, method_func in methods_to_analyze.items():
            honest_winner, _, _, _ = method_func(agents_df, candidates, tie_breaker_method)
            strategic_winner, _, _, _ = method_func(strategic_df, candidates, tie_breaker_method)
            
            results_accumulator.append({
                "method": name, "honest_winner": honest_winner, "strategic_winner": strategic_winner,
                "condorcet_winner": condorcet_winner, "utilitarian_winner": utilitarian_winner,
                "majority_winner": majority_winner, "plurality_winner": plurality_winner
            })
        
        if i == num_simulations - 1:
            last_run_df, last_run_candidates = agents_df, candidates

        progress_bar.progress((i + 1) / num_simulations, text=f"Simulação {i+1}/{num_simulations} concluída.")

    summary_df = pd.DataFrame(results_accumulator)
    analysis_results = []
    for method in methods_to_analyze.keys():
        method_df = summary_df[summary_df['method'] == method]
        
        num_total_sims = len(method_df)
        num_majority_cases = method_df['majority_winner'].notna().sum()
        num_plurality_cases = method_df['plurality_winner'].notna().sum()

        justice_freq = (method_df['honest_winner'] == method_df['condorcet_winner']).sum() / num_total_sims * 100
        satisfaction_freq = (method_df['honest_winner'] == method_df['utilitarian_winner']).sum() / num_total_sims * 100
        resilience_freq = (method_df['honest_winner'] == method_df['strategic_winner']).mean() * 100
        
        majority_freq = (method_df['honest_winner'] == method_df['majority_winner']).sum() / num_majority_cases * 100 if num_majority_cases > 0 else 100
        
        plurality_agreement_freq = (method_df['honest_winner'] == method_df['plurality_winner']).sum() / num_plurality_cases * 100 if num_plurality_cases > 0 else 100

        failure_freq = ((method_df['honest_winner'] == "Anulada") | (method_df['honest_winner'] == "Paradoxo") | (method_df['honest_winner'] == "N/A")).mean() * 100

        final_score = (justice_freq * 0.4) + (satisfaction_freq * 0.2) + (majority_freq * 0.2) + (resilience_freq * 0.1) - (failure_freq * 0.1)

        analysis_results.append({
            "Método": method, "Score Final": final_score,
            "Vence Todos 1x1 (%)": justice_freq,
            "Elege o com Maior Média (%)": satisfaction_freq,
            "Elege Quando Há >50% (%)": majority_freq,
            "Alinhado ao Voto Simples (%)": plurality_agreement_freq,
            "Resistente à Estratégia (%)": resilience_freq,
            "Falhas (Empate/Paradoxo) (%)": failure_freq
        })

    final_df = pd.DataFrame(analysis_results)
    final_df["Rank"] = final_df["Score Final"].rank(ascending=False, method="min").astype(int)
    final_df = final_df.sort_values("Rank")

    # Metadados dos métodos de votação
    methods_metadata = {
        "Pluralidade": {"Descrição": "O candidato com mais votos de 1ª preferência vence. Simples, mas vulnerável ao 'voto útil'.", "Prós": "Simples, fácil de entender e implementar.", "Contras": "Pode eleger um candidato impopular, incentiva o voto útil, não reflete a intensidade da preferência.", "Quando Usar": "Eleições rápidas, pequenas comunidades, onde a simplicidade é primordial.", "Turnos": "1"},
        "Dois Turnos (Runoff)": {"Descrição": "Se ninguém tem maioria no 1º turno, os 2 mais votados vão para um 2º turno. Garante maioria.", "Prós": "Garante que o vencedor tenha apoio da maioria absoluta, reduz o voto útil.", "Contras": "Mais caro e demorado, pode eliminar candidatos populares no 1º turno.", "Quando Usar": "Eleições presidenciais, prefeituras, onde a legitimidade da maioria é crucial.", "Turnos": "2"},
        "Voto Ranqueado (IRV)": {"Descrição": "Eleitores ranqueiam os candidatos. O menos votado é eliminado e seus votos são redistribuídos até um candidato ter maioria.", "Prós": "Garante maioria, reduz o voto útil, elege candidatos de consenso.", "Contras": "Complexo para eleitores e apuração, pode ter resultados contraintuitivos.", "Quando Usar": "Eleições parlamentares, internas de partidos, onde a representatividade é valorizada.", "Turnos": "Variável"},
        "Voto por Aprovação": {"Descrição": "Eleitores podem 'aprovar' (votar em) quantos candidatos quiserem (preferência > 0.5). O mais aprovado vence.", "Prós": "Simples, expressa apoio amplo, reduz o voto útil, elege candidatos de consenso.", "Contras": "Não reflete a ordem de preferência, pode ser manipulado por 'bullet voting'.", "Quando Usar": "Eleições internas, conselhos, onde a aceitação geral é importante.", "Turnos": "1"},
        "Voto por Pontuação": {"Descrição": "Eleitores dão uma nota (0 a 10) a cada candidato. O de maior nota média vence.", "Prós": "Expressa intensidade da preferência, elege candidatos de consenso, resistente a manipulação.", "Contras": "Pode ser complexo, vulnerável a 'burying' (rebaixamento estratégico).", "Quando Usar": "Pesquisas de opinião, avaliações, onde a intensidade da opinião é relevante.", "Turnos": "1"},
        "Contagem de Borda": {"Descrição": "Candidatos recebem pontos baseados em sua posição no ranking de cada eleitor. Tende a eleger candidatos de consenso.", "Prós": "Elege candidatos de consenso, reflete o ranking completo, menos polarizador.", "Contras": "Vulnerável a manipulação (rebaixamento estratégico), pode não eleger o vencedor Condorcet.", "Quando Usar": "Eleições em grupos pequenos, comitês, onde o consenso é valorizado.", "Turnos": "1"},
        "Voto Contingente": {"Descrição": "Versão simplificada do IRV. Se ninguém tem maioria, apenas os 2 primeiros sobrevivem e recebem os votos dos eliminados.", "Prós": "Mais simples que IRV, garante maioria, reduz o voto útil.", "Contras": "Pode eliminar candidatos populares no 1º turno, menos preciso que IRV.", "Quando Usar": "Eleições onde a simplicidade e a maioria são importantes, mas IRV é muito complexo.", "Turnos": "2"},
        "Condorcet": {"Descrição": "O vencedor é aquele que venceria todos os outros em confrontos diretos (um-contra-um).", "Prós": "Considerado o mais 'justo' e reflete a vontade da maioria em duelos diretos.", "Contras": "Pode não haver um vencedor (paradoxo), complexo para eleitores e apuração.", "Quando Usar": "Análise teórica, como benchmark para outros sistemas.", "Turnos": "Variável (duelos)"},
        "Método de Copeland": {"Descrição": "Solução para o paradoxo de Condorcet. O vencedor é o que vence mais confrontos diretos.", "Prós": "Sempre produz um vencedor, baseado no critério Condorcet.", "Contras": "Complexo, pode não ser intuitivo, ainda pode ter empates.", "Quando Usar": "Quando um vencedor Condorcet é desejado, mas a garantia de um resultado é necessária.", "Turnos": "Variável (duelos)"},
        "Anti-Pluralidade": {"Descrição": "Eleitores votam no candidato que *menos* querem. O com menos votos 'contra' vence.", "Prós": "Simples, pode evitar a eleição de candidatos amplamente rejeitados.", "Contras": "Não reflete a preferência positiva, pode ser contraintuitivo.", "Quando Usar": "Análise teórica, para ilustrar o voto de rejeição.", "Turnos": "1"}
    }
    desc_df = pd.DataFrame.from_dict(methods_metadata, orient='index')
    desc_df.index.name = "Método"
    desc_df = desc_df.reset_index()

    # Combina os resultados com as descrições
    merged_df = pd.merge(final_df, desc_df, on="Método")
    
    # Reordena as colunas
    cols = ["Rank", "Método", "Score Final", "Vence Todos 1x1 (%)", "Elege o com Maior Média (%)", "Elege Quando Há >50% (%)", "Alinhado ao Voto Simples (%)", "Resistente à Estratégia (%)", "Falhas (Empate/Paradoxo) (%)"]
    desc_cols = ["Descrição", "Prós", "Contras", "Quando Usar", "Turnos"]
    merged_df = merged_df[cols + desc_cols]

    st.session_state.last_analysis_summary = merged_df
    st.session_state.last_example_df = last_run_df
    st.session_state.last_example_candidates = last_run_candidates

if st.session_state.last_analysis_summary is not None:
    st.info("A tabela classifica os sistemas por um 'Score Final'. Clique em uma linha para ver a análise detalhada de um cenário da última simulação.")
    
    summary_df = st.session_state.last_analysis_summary
    
    formatter = {
        "Score Final": "{:.1f}", "Vence Todos 1x1 (%)": "{:.1f}%", "Elege o com Maior Média (%)": "{:.1f}%",
        "Elege Quando Há >50% (%)": "{:.1f}%", "Alinhado ao Voto Simples (%)": "{:.1f}%",
        "Resistente à Estratégia (%)": "{:.1f}%", "Falhas (Empate/Paradoxo) (%)": "{:.1f}%"
    }
    
    # DataFrame interativo principal
    selected_rows = st.dataframe(
        summary_df.style.format(formatter).bar(subset=["Score Final"], vmin=0, vmax=100, color='#5fba7d'),
        on_select='rerun',
        selection_mode='single-row',
        hide_index=True,
        use_container_width=True,
        key='system_selector'
    )
    
    st.markdown("---")
    st.header("Análise Detalhada da Última Simulação")
    st.warning("Os dados abaixo são da **última** simulação executada. Use a tabela acima para selecionar um sistema e ver os detalhes.")

    # Lógica de seleção
    example_method_name = summary_df.iloc[0]['Método'] # Default para o primeiro do rank
    if selected_rows and selected_rows['selection']['rows']:
        selected_index = selected_rows['selection']['rows'][0]
        example_method_name = summary_df.loc[selected_index, 'Método']

    example_voting_func = methods_to_analyze[example_method_name]
    
    example_df = st.session_state.last_example_df
    example_candidates = st.session_state.last_example_candidates
    strategic_df_example = apply_strategic_vote(example_df, example_candidates, strategic_prob)
    honest_winner, _, _, _ = example_voting_func(example_df, example_candidates, tie_breaker_method)
    strategic_winner, results, tie_occurred, declared_votes = example_voting_func(strategic_df_example, example_candidates, tie_breaker_method)
    
    if not example_df.equals(strategic_df_example):
        changed_indices = (example_df['rank_1'] != strategic_df_example['rank_1'])
        vote_type = pd.Series("Natural", index=example_df.index)
        vote_type[changed_indices] = "Estratégico"
    else:
        vote_type = pd.Series("Natural", index=example_df.index)

    plot_title = "Preferências Naturais dos Agentes (Última simulação)"
    col1, col2, col3 = st.columns([2, 3, 1.5])
    with col1:
        st.subheader("Visualização")
        if len(example_candidates) == 2:
            fig = create_2d_plot(example_df, example_candidates, plot_title)
        elif len(example_candidates) == 3:
            fig = create_3d_plot(example_df, example_candidates, plot_title)
        else:
            fig = create_parallel_coordinates_plot(example_df, example_candidates, plot_title)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Agentes (Última simulação)")
        pref_cols = [f"pref_{c}" for c in example_candidates]
        display_df = example_df.copy()
        
        display_df['Voto Natural'] = display_df['rank_1']
        display_df['Voto Declarado'] = declared_votes.astype(str)
        display_df['Tipo de Voto'] = vote_type
        format_dict = {'Convicção (Desvio Padrão das Prefs.)': "{:.3f}"}
        for col in pref_cols: format_dict[col] = "{:.2f}"
        st.dataframe(display_df.style.background_gradient(cmap='viridis', subset=pref_cols).format(format_dict), use_container_width=True)
    with col3:
        st.subheader(f"Resultados (Última simulação)")
        st.metric("Vencedor", strategic_winner)
        if tie_occurred and tie_breaker_method != "Anulação da Votação":
            st.info(f"Desempate: {tie_breaker_method}")
        if strategic_prob > 0:
            if honest_winner == strategic_winner:
                st.success(f"Estratégia não alterou o vencedor.")
            else:
                st.warning(f"Estratégia alterou: {honest_winner} ➡️ {strategic_winner}")
        st.text("Detalhes da Votação:")
        if example_method_name == "Voto Ranqueado (IRV)":
            for i, round_counts in enumerate(results):
                st.text(f"Rodada {i+1}:")
                st.dataframe(round_counts, use_container_width=True)
        elif example_method_name in ["Dois Turnos (Runoff)", "Voto Contingente"]:
            if "Resultado Final" in results:
                st.text("Preferências de 1º Turno:")
                st.dataframe(results["Turno 1"], use_container_width=True)
                st.text("Resultado Final:")
                st.dataframe(results["Resultado Final"], use_container_width=True)
            elif "Turno 2" in results:
                st.text("Turno 1:")
                st.dataframe(results["Turno 1"], use_container_width=True)
                st.text(f"Finalistas: {results['Finalistas'][0]} vs {results['Finalistas'][1]}")
                st.text("Turno 2:")
                st.dataframe(results["Turno 2"], use_container_width=True)
            else:
                st.text("Turno 1 (Vencedor por maioria):")
                st.dataframe(results["Turno 1"], use_container_width=True)
        else:
            st.dataframe(results, use_container_width=True)
else:
    st.info("Ajuste os parâmetros na barra lateral e clique em 'Executar Análise Estatística' para começar.")