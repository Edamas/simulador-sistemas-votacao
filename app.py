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
    votes_df = df[rank_cols].copy()
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
        for idx, row in votes_df.iterrows():
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

# Inicializa o estado da sessão
if 'analysis_results_df' not in st.session_state:
    st.session_state.analysis_results_df = None
    st.session_state.analysis_desc_df = None
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
    final_df = final_df.sort_values("Rank").reset_index(drop=True)
    cols = ["Rank", "Método", "Score Final", "Vence Todos 1x1 (%)", "Elege o com Maior Média (%)", "Elege Quando Há >50% (%)", "Alinhado ao Voto Simples (%)", "Resistente à Estratégia (%)", "Falhas (Empate/Paradoxo) (%)"]
    final_df = final_df[cols]

    methods_metadata = {
        "Pluralidade": {
            "Descrição": "Também conhecido como 'First-Past-the-Post', é o sistema mais simples. Cada eleitor escolhe **apenas um** candidato, e o que tiver mais votos vence, mesmo sem maioria absoluta.",
            "Exemplo": "Numa eleição com 100 eleitores para os candidatos A, B e C:\n- A recebe 40 votos.\n- B recebe 35 votos.\n- C recebe 25 votos.\n**Resultado:** A vence com 40% dos votos, embora 60% dos eleitores preferissem outro candidato.",
            "Prós": "Simples de entender e apurar.",
            "Contras": "Incentiva o 'voto útil' e pode eleger um candidato que a maioria rejeita."
        },
        "Dois Turnos (Runoff)": {
            "Descrição": "Se nenhum candidato atinge a maioria absoluta (>50%) no primeiro turno, os **dois mais votados** avançam para um segundo turno. Isso garante que o vencedor tenha o apoio da maioria.",
            "Exemplo": "Eleição com 100 eleitores (A, B, C):\n- **1º Turno:** A (40), B (35), C (25).\n- Ninguém tem >50%, então A e B vão para o 2º turno.\n- **2º Turno:** Os eleitores de C agora votam em A ou B. Se a maioria deles preferir B, o resultado pode ser B (35+15=50) vs A (40+10=50), levando a um empate, ou um deles vencer.",
            "Prós": "Garante um vencedor com maioria absoluta, dando-lhe mais legitimidade.",
            "Contras": "Mais caro e demorado. Candidatos de consenso, mas menos populares, podem ser eliminados no 1º turno."
        },
        "Voto Ranqueado (IRV)": {
            "Descrição": "Os eleitores **ranqueiam** os candidatos em ordem de preferência. Se ninguém tem maioria, o candidato com menos votos de 1ª preferência é eliminado. Seus votos são redistribuídos para a próxima preferência de cada eleitor. O processo se repete até um candidato ter maioria.",
            "Exemplo": "100 eleitores (A, B, C):\n- **Rodada 1:** A (40), B (35), C (25). C é eliminado.\n- **Rodada 2:** Os 25 votos de C são transferidos. Se 15 deles tinham B como 2ª opção e 10 tinham A:\n  - A: 40 + 10 = 50\n  - B: 35 + 15 = 50\n**Resultado:** Empate entre A e B. O IRV busca um vencedor de consenso.",
            "Prós": "Reduz o 'voto útil' e permite que eleitores votem em quem realmente preferem. Elege vencedores de maior consenso.",
            "Contras": "Apuração complexa e pode ter resultados não intuitivos (ex: um candidato pode vencer mesmo que outro fosse preferido contra ele em um confronto direto)."
        },
        "Voto por Aprovação": {
            "Descrição": "Eleitores podem votar em (ou 'aprovar') **quantos candidatos quiserem**. O candidato com o maior número de aprovações vence. É como uma eleição de múltipla escolha.",
            "Exemplo": "Numa eleição com 3 eleitores:\n- Eleitor 1 aprova A e B.\n- Eleitor 2 aprova B.\n- Eleitor 3 aprova A, B e C.\n**Resultado:** A (2 votos), B (3 votos), C (1 voto). B vence.",
            "Prós": "Simples, expressa apoio amplo e tende a eleger candidatos de menor rejeição.",
            "Contras": "Não permite expressar uma ordem de preferência. Um eleitor que aprova seu favorito e um candidato 'aceitável' dá o mesmo peso a ambos."
        },
        "Voto por Pontuação": {
            "Descrição": "Eleitores dão uma **nota** (ex: 0 a 10) para cada candidato. O candidato com a maior nota média (ou soma total) vence. Permite expressar a intensidade da preferência.",
            "Exemplo": "2 eleitores, 3 candidatos (notas de 0 a 5):\n- Eleitor 1: A(5), B(1), C(0).\n- Eleitor 2: A(2), B(5), C(4).\n**Resultado:** Soma A=7, B=6, C=4. A vence.",
            "Prós": "Captura a intensidade da preferência e promove candidatos de alto consenso.",
            "Contras": "Vulnerável a estratégias de 'rebaixamento', onde eleitores dão nota mínima a concorrentes fortes para ajudar seu favorito."
        },
        "Contagem de Borda": {
            "Descrição": "Eleitores ranqueiam os candidatos. Cada posição no ranking vale pontos (ex: 1º lugar = N-1 pontos, 2º = N-2, etc., onde N é o nº de candidatos). O candidato com mais pontos vence.",
            "Exemplo": "3 candidatos (A,B,C). 1º=2pts, 2º=1pt, 3º=0pts. 5 eleitores:\n- 3 eleitores: (A > B > C) -> A: 3*2=6, B: 3*1=3\n- 2 eleitores: (B > C > A) -> B: 2*2=4, C: 2*1=2\n**Resultado:** A=6, B=7, C=2. B vence, pois foi bem ranqueado por todos.",
            "Prós": "Elege candidatos de consenso, que podem não ser o favorito da maioria, mas são amplamente aceitáveis.",
            "Contras": "Vulnerável à clonagem de candidatos (candidatos semelhantes podem dividir os pontos de um oponente forte) e ao 'rebaixamento' estratégico."
        },
        "Voto Contingente": {
            "Descrição": "Uma versão simplificada do IRV. Se ninguém tem maioria, **todos os candidatos são eliminados, exceto os dois primeiros**. Os votos dos eleitores dos candidatos eliminados são então transferidos para um dos dois finalistas, conforme a preferência.",
            "Exemplo": "100 eleitores (A, B, C, D):\n- **1º Turno:** A(35), B(30), C(20), D(15). Ninguém tem >50%.
- **Contingência:** A e B avançam. Os votos de C e D são transferidos para A ou B. Se a maioria dos eleitores de C e D preferem B a A, B pode vencer.",
            "Prós": "Mais simples que o IRV, mas ainda garante um vencedor com maioria.",
            "Contras": "Pode eliminar um 'candidato de consenso' no primeiro turno (ex: um candidato que seria a 2ª opção de todos)."
        },
        "Condorcet": {
            "Descrição": "Um método teórico que define o vencedor 'ideal'. O vencedor Condorcet é o candidato que, em **confrontos diretos (um-contra-um)**, venceria todos os outros candidatos.",
            "Exemplo": "Candidatos A, B, C. A maioria prefere A a B, A a C, e B a C.\n- A vs B -> A vence.\n- A vs C -> A vence.\n- B vs C -> B vence.\n**Resultado:** A é o vencedor Condorcet.",
            "Prós": "Considerado o critério mais 'justo' de uma eleição.",
            "Contras": "Pode não haver um vencedor (Paradoxo de Condorcet, ex: A>B, B>C, C>A), tornando-o mais um critério de avaliação do que um método prático."
        },
        "Método de Copeland": {
            "Descrição": "Uma forma de encontrar um vencedor usando o critério Condorcet. O vencedor é o candidato que **vence o maior número de confrontos diretos** (um-contra-um).",
            "Exemplo": "A vs B (A vence), A vs C (A vence), B vs C (B vence).\n- **Placar:** A (2 vitórias), B (1 vitória), C (0 vitórias).\n**Resultado:** A vence.",
            "Prós": "Sempre produz um resultado e é baseado no 'justo' critério Condorcet.",
            "Contras": "Pode resultar em empates e a apuração é complexa."
        },
        "Anti-Pluralidade": {
            "Descrição": "Também conhecido como 'Voto de Rejeição'. Cada eleitor vota no candidato que **menos** deseja. O candidato com o menor número de votos 'contra' é o vencedor.",
            "Exemplo": "100 eleitores votam para rejeitar A, B ou C:\n- A recebe 10 votos 'contra'.\n- B recebe 30 votos 'contra'.\n- C recebe 60 votos 'contra'.\n**Resultado:** A vence, por ser o menos rejeitado.",
            "Prós": "Simples e eficaz para evitar a eleição de um candidato amplamente impopular.",
            "Contras": "Não expressa preferência positiva e pode eleger um candidato que poucos realmente apoiam."
        }
    }
    desc_df = pd.DataFrame.from_dict(methods_metadata, orient='index')
    desc_df.index.name = "Método"
    desc_df = desc_df.reset_index()

    st.session_state.analysis_results_df = final_df
    st.session_state.analysis_desc_df = desc_df
    st.session_state.last_example_df = last_run_df
    st.session_state.last_example_candidates = last_run_candidates

if st.session_state.get('analysis_results_df') is not None:
    st.info("A tabela abaixo classifica os sistemas por um 'Score Final'. **Clique em uma linha** para ver a descrição detalhada e a análise do sistema selecionado.")
    
    results_df = st.session_state.analysis_results_df
    desc_df = st.session_state.analysis_desc_df
    
    formatter = {
        "Score Final": "{:.1f}", "Vence Todos 1x1 (%)": "{:.1f}%", "Elege o com Maior Média (%)": "{:.1f}%",
        "Elege Quando Há >50% (%)": "{:.1f}%", "Alinhado ao Voto Simples (%)": "{:.1f}%",
        "Resistente à Estratégia (%)": "{:.1f}%", "Falhas (Empate/Paradoxo) (%)": "{:.1f}%"
    }
    
    selection = st.dataframe(
        results_df.style.format(formatter).bar(subset=["Score Final"], vmin=0, vmax=100, color='#5fba7d'),
        on_select='rerun',
        selection_mode='single-row',
        hide_index=True,
        use_container_width=True,
        key='system_selector'
    )
    
    selected_method_name = results_df.iloc[0]['Método']
    if selection['selection']['rows']:
        selected_index = selection['selection']['rows'][0]
        selected_method_name = results_df.loc[selected_index, 'Método']

    st.markdown("---")
    
    selected_method_info = desc_df[desc_df['Método'] == selected_method_name].iloc[0]
    
    with st.container(border=True):
        st.header(f"Como funciona: {selected_method_name}")
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.subheader("Descrição")
            st.markdown(selected_method_info['Descrição'].replace('\n', '\n\n'))
            st.subheader("Exemplo Prático")
            st.markdown(selected_method_info['Exemplo'].replace('\n', '\n\n'))
        with col2:
            st.subheader("Prós")
            st.success(selected_method_info['Prós'])
            st.subheader("Contras")
            st.error(selected_method_info['Contras'])

    st.markdown("---")
    st.header(f"Análise da Última Simulação para '{selected_method_name}'")
    st.warning("Os dados abaixo são da **última** simulação executada, aplicando o sistema selecionado acima.")

    example_voting_func = methods_to_analyze[selected_method_name]
    
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
        if selected_method_name == "Voto Ranqueado (IRV)":
            for i, round_counts in enumerate(results):
                st.text(f"Rodada {i+1}:")
                st.dataframe(round_counts, use_container_width=True)
        elif selected_method_name in ["Dois Turnos (Runoff)", "Voto Contingente"]:
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
