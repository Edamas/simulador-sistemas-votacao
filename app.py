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
        return sorted(prefs, key=lambda k: prefs[k], reverse=True)

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

# --- Funções dos Métodos de Votação ---
def run_plurality(df, candidates, tie_breaker_method):
    counts = df['rank_1'].value_counts()
    max_votes = counts.max()
    tied = counts[counts == max_votes].index.tolist()
    winner = break_tie(tied, tie_breaker_method) if len(tied) > 1 else tied[0]
    return winner, counts, len(tied) > 1, df['rank_1']

def run_two_round_runoff_top2(df, candidates, tie_breaker_method):
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
    if len(top_2) < 2:
        return counts_r1.index[0], {"Turno 1": counts_r1}, False, df['rank_1']
    round_2_votes = df.apply(lambda row: top_2[0] if row[f'pref_{top_2[0]}'] > row[f'pref_{top_2[1]}'] else top_2[1], axis=1)
    counts_r2 = round_2_votes.value_counts()
    max_votes_r2 = counts_r2.max()
    tied_r2 = counts_r2[counts_r2 == max_votes_r2].index.tolist()
    winner = break_tie(tied_r2, tie_breaker_method) if len(tied_r2) > 1 else tied_r2[0]
    results = {"Turno 1": counts_r1, "Finalistas": top_2, "Turno 2": counts_r2}
    return winner, results, len(tied_r2) > 1, round_2_votes

def run_two_round_runoff_top3(df, candidates, tie_breaker_method):
    num_voters = len(df)
    counts_r1 = df['rank_1'].value_counts()
    if counts_r1.empty:
        return "N/A", {}, False, pd.Series()
    if counts_r1.max() > num_voters / 2:
        max_votes = counts_r1.max()
        tied = counts_r1[counts_r1 == max_votes].index.tolist()
        winner = break_tie(tied, tie_breaker_method) if len(tied) > 1 else tied[0]
        return winner, {"Turno 1": counts_r1}, len(tied) > 1, df['rank_1']
    top_3 = counts_r1.index[:3].tolist()
    if len(top_3) < 3:
        return counts_r1.index[0], {"Turno 1": counts_r1}, False, df['rank_1']
    def get_r2_vote(row):
        prefs = {c: row[f'pref_{c}'] for c in top_3}
        return max(prefs, key=lambda k: prefs[k])
    round_2_votes = df.apply(get_r2_vote, axis=1)
    counts_r2 = round_2_votes.value_counts()
    max_votes_r2 = counts_r2.max()
    tied_r2 = counts_r2[counts_r2 == max_votes_r2].index.tolist()
    winner = break_tie(tied_r2, tie_breaker_method) if len(tied_r2) > 1 else tied_r2[0]
    results = {"Turno 1": counts_r1, "Finalistas": top_3, "Turno 2": counts_r2}
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

def run_approval_free(df, candidates, tie_breaker_method):
    votes = df.apply(lambda row: tuple(sorted([c for c in candidates if row[f'pref_{c}'] > 0.5])), axis=1)
    all_approvals = Counter(c for vote_tuple in votes for c in vote_tuple)
    if not all_approvals: return "Ninguém", pd.Series(), False, pd.Series([()]*len(df))
    counts = pd.Series(all_approvals).sort_values(ascending=False)
    max_approvals = counts.max()
    tied = counts[counts == max_approvals].index.tolist()
    winner = break_tie(tied, tie_breaker_method) if len(tied) > 1 else tied[0]
    return winner, counts, len(tied) > 1, votes

def run_approval_fixed(df, candidates, tie_breaker_method, num_fixed_votes):
    all_approvals = Counter()
    declared_votes_list = []
    for idx, row in df.iterrows():
        prefs = {c: row[f'pref_{c}'] for c in candidates}
        sorted_candidates = sorted(prefs, key=lambda k: prefs[k], reverse=True)
        approved_candidates = sorted_candidates[:num_fixed_votes]
        for c in approved_candidates:
            all_approvals[c] += 1
        declared_votes_list.append(tuple(sorted(approved_candidates)))
    if not all_approvals: return "Ninguém", pd.Series(), False, pd.Series([()]*len(df))
    counts = pd.Series(all_approvals).sort_values(ascending=False)
    max_approvals = counts.max()
    tied = counts[counts == max_approvals].index.tolist()
    winner = break_tie(tied, tie_breaker_method) if len(tied) > 1 else tied[0]
    return winner, counts, len(tied) > 1, pd.Series(declared_votes_list)

def run_approval_1_fixed(df, candidates, tie_breaker_method): return run_approval_fixed(df, candidates, tie_breaker_method, 1)
def run_approval_2_fixed(df, candidates, tie_breaker_method): return run_approval_fixed(df, candidates, tie_breaker_method, 2)
def run_approval_3_fixed(df, candidates, tie_breaker_method): return run_approval_fixed(df, candidates, tie_breaker_method, 3)
def run_approval_4_fixed(df, candidates, tie_breaker_method): return run_approval_fixed(df, candidates, tie_breaker_method, 4)
def run_approval_5_fixed(df, candidates, tie_breaker_method): return run_approval_fixed(df, candidates, tie_breaker_method, 5)
def run_approval_6_fixed(df, candidates, tie_breaker_method): return run_approval_fixed(df, candidates, tie_breaker_method, 6)
def run_approval_7_fixed(df, candidates, tie_breaker_method): return run_approval_fixed(df, candidates, tie_breaker_method, 7)
def run_approval_8_fixed(df, candidates, tie_breaker_method): return run_approval_fixed(df, candidates, tie_breaker_method, 8)
def run_approval_9_fixed(df, candidates, tie_breaker_method): return run_approval_fixed(df, candidates, tie_breaker_method, 9)

def run_score_n_levels(df, candidates, tie_breaker_method, max_score_value):
    scores = pd.DataFrame()
    for c in candidates:
        if max_score_value == 10:
            scores[c] = (df[f'pref_{c}'] * 9 + 1).round()
        else:
            scores[c] = (df[f'pref_{c}'] * max_score_value).round()
    total_scores = scores.sum().sort_values(ascending=False)
    max_score = total_scores.max()
    tied = total_scores[total_scores == max_score].index.tolist()
    winner = break_tie(tied, tie_breaker_method) if len(tied) > 1 else tied[0]
    return winner, total_scores, len(tied) > 1, df['rank_1']

def run_score_n_levels_normalized(df, candidates, tie_breaker_method, max_score_value):
    scores = pd.DataFrame()
    for c in candidates:
        if max_score_value == 10:
            scores[c] = (df[f'pref_{c}'] * 9 + 1)
        else:
            scores[c] = (df[f'pref_{c}'] * max_score_value)

    min_score = 1 if max_score_value == 10 else 0
    scores_shifted = scores - min_score
    row_sums = scores_shifted.sum(axis=1)
    row_sums[row_sums == 0] = 1
    normalized_scores = scores_shifted.div(row_sums, axis=0)

    total_scores = normalized_scores.sum().sort_values(ascending=False)
    max_score = total_scores.max()
    tied = total_scores[total_scores == max_score].index.tolist()
    winner = break_tie(tied, tie_breaker_method) if len(tied) > 1 else tied[0]
    return winner, total_scores, len(tied) > 1, df['rank_1']

def run_score_10_levels(df, candidates, tie_breaker_method): return run_score_n_levels(df, candidates, tie_breaker_method, 10)
def run_score_3_levels(df, candidates, tie_breaker_method): return run_score_n_levels(df, candidates, tie_breaker_method, 2)
def run_score_4_levels(df, candidates, tie_breaker_method): return run_score_n_levels(df, candidates, tie_breaker_method, 3)
def run_score_5_levels(df, candidates, tie_breaker_method): return run_score_n_levels(df, candidates, tie_breaker_method, 4)
def run_score_6_levels(df, candidates, tie_breaker_method): return run_score_n_levels(df, candidates, tie_breaker_method, 5)
def run_score_7_levels(df, candidates, tie_breaker_method): return run_score_n_levels(df, candidates, tie_breaker_method, 6)
def run_score_2_levels(df, candidates, tie_breaker_method): return run_score_n_levels(df, candidates, tie_breaker_method, 1)

def run_score_10_levels_normalized(df, candidates, tie_breaker_method): return run_score_n_levels_normalized(df, candidates, tie_breaker_method, 10)
def run_score_2_levels_normalized(df, candidates, tie_breaker_method): return run_score_n_levels_normalized(df, candidates, tie_breaker_method, 1)
def run_score_3_levels_normalized(df, candidates, tie_breaker_method): return run_score_n_levels_normalized(df, candidates, tie_breaker_method, 2)
def run_score_4_levels_normalized(df, candidates, tie_breaker_method): return run_score_n_levels_normalized(df, candidates, tie_breaker_method, 3)
def run_score_5_levels_normalized(df, candidates, tie_breaker_method): return run_score_n_levels_normalized(df, candidates, tie_breaker_method, 4)
def run_score_6_levels_normalized(df, candidates, tie_breaker_method): return run_score_n_levels_normalized(df, candidates, tie_breaker_method, 5)
def run_score_7_levels_normalized(df, candidates, tie_breaker_method): return run_score_n_levels_normalized(df, candidates, tie_breaker_method, 6)

def run_borda_classic(df, candidates, tie_breaker_method):
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

def run_borda_dowdall(df, candidates, tie_breaker_method):
    points = {c: 0 for c in candidates}
    num_candidates = len(candidates)
    for i in range(num_candidates):
        score = 1 / (i + 1)
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
    margins = {c: 0 for c in candidates}
    for c1, c2 in pairs:
        c1_votes = (df[f'pref_{c1}'] > df[f'pref_{c2}']).sum()
        c2_votes = (df[f'pref_{c2}'] > df[f'pref_{c1}']).sum()
        if c1_votes > c2_votes:
            wins[c1] += 1
            margins[c1] += c1_votes - c2_votes
            margins[c2] -= c1_votes - c2_votes
        elif c2_votes > c1_votes:
            wins[c2] += 1
            margins[c2] += c2_votes - c1_votes
            margins[c1] -= c2_votes - c1_votes
    return wins, margins

def run_condorcet_leveled_score(df, candidates, tie_breaker_method, levels, normalized=False):
    pairs = list(combinations(candidates, 2))
    
    pairwise_scores = pd.DataFrame()
    scores = pd.Series(dtype=float)
    for c1, c2 in pairs:
        diff = df[f'pref_{c1}'] - df[f'pref_{c2}']
        
        if levels == 2: # Binary
            scores = pd.Series(0, index=df.index)
            scores[diff > 0] = 1
            scores[diff < 0] = -1
        elif levels == 3:
            scores = pd.Series(0, index=df.index)
            scores[diff > 1/3] = 1
            scores[diff < -1/3] = -1
        elif levels == 4:
            scores = pd.Series(0, index=df.index)
            scores[diff > 0.5] = 2
            scores[(diff > 0) & (diff <= 0.5)] = 1
            scores[diff < -0.5] = -2
            scores[(diff < 0) & (diff >= -0.5)] = -1
        elif levels == 5:
            scores = pd.Series(0, index=df.index)
            scores[diff > 0.6] = 2
            scores[(diff > 0.2) & (diff <= 0.6)] = 1
            scores[diff < -0.6] = -2
            scores[(diff < -0.2) & (diff >= -0.6)] = -1
        
        pairwise_scores[f'{c1}_vs_{c2}'] = scores

    if normalized:
        row_sums = pairwise_scores.abs().sum(axis=1)
        row_sums[row_sums == 0] = 1
        pairwise_scores = pairwise_scores.div(row_sums, axis=0)

    final_scores = {c: 0 for c in candidates}
    for c1, c2 in pairs:
        margin = pairwise_scores[f'{c1}_vs_{c2}'].sum()
        final_scores[c1] += margin
        final_scores[c2] -= margin
        
    counts = pd.Series(final_scores).sort_values(ascending=False)
    max_score = counts.max()
    tied = counts[counts == max_score].index.tolist()
    winner = break_tie(tied, tie_breaker_method) if len(tied) > 1 else tied[0]
    
    return winner, counts, len(tied) > 1, df['rank_1']

def run_condorcet_2_levels(df, candidates, tie_breaker_method):
    return run_condorcet_leveled_score(df, candidates, tie_breaker_method, 2, normalized=False)

def run_condorcet_3_levels(df, candidates, tie_breaker_method):
    return run_condorcet_leveled_score(df, candidates, tie_breaker_method, 3, normalized=False)

def run_condorcet_4_levels(df, candidates, tie_breaker_method):
    return run_condorcet_leveled_score(df, candidates, tie_breaker_method, 4, normalized=False)

def run_condorcet_5_levels(df, candidates, tie_breaker_method):
    return run_condorcet_leveled_score(df, candidates, tie_breaker_method, 5, normalized=False)

def run_condorcet_2_levels_normalized(df, candidates, tie_breaker_method):
    return run_condorcet_leveled_score(df, candidates, tie_breaker_method, 2, normalized=True)

def run_condorcet_3_levels_normalized(df, candidates, tie_breaker_method):
    return run_condorcet_leveled_score(df, candidates, tie_breaker_method, 3, normalized=True)

def run_condorcet_4_levels_normalized(df, candidates, tie_breaker_method):
    return run_condorcet_leveled_score(df, candidates, tie_breaker_method, 4, normalized=True)

def run_condorcet_5_levels_normalized(df, candidates, tie_breaker_method):
    return run_condorcet_leveled_score(df, candidates, tie_breaker_method, 5, normalized=True)

def run_condorcet(df, candidates, tie_breaker_method):
    wins, _ = run_condorcet_base(df, candidates)
    for candidate, num_wins in wins.items():
        if num_wins == len(candidates) - 1:
            return candidate, pd.Series(wins), False, df['rank_1']
    return "Paradoxo", pd.Series(wins), True, df['rank_1']

def run_copeland_wins(df, candidates, tie_breaker_method):
    wins, _ = run_condorcet_base(df, candidates)
    counts = pd.Series(wins).sort_values(ascending=False)
    max_wins = counts.max()
    tied = counts[counts == max_wins].index.tolist()
    winner = break_tie(tied, tie_breaker_method) if len(tied) > 1 else tied[0]
    return winner, counts, len(tied) > 1, df['rank_1']

def run_copeland_margin(df, candidates, tie_breaker_method):
    _, margins = run_condorcet_base(df, candidates)
    counts = pd.Series(margins).sort_values(ascending=False)
    max_margin = counts.max()
    tied = counts[counts == max_margin].index.tolist()
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
    contingent_votes_series = df['rank_1'].copy()
    eliminated_voters = contingent_votes_series[~contingent_votes_series.isin(top_2)].index
    for idx in eliminated_voters:
        if df.loc[idx, f'pref_{top_2[0]}'] > df.loc[idx, f'pref_{top_2[1]}']:
            contingent_votes_series.loc[idx] = top_2[0]
        else:
            contingent_votes_series.loc[idx] = top_2[1]
    final_counts = contingent_votes_series.value_counts()
    if final_counts.empty:
        return "Anulada", {"Turno 1": counts_r1, "Resultado Final": final_counts}, True, contingent_votes_series
    max_final_votes = final_counts.max()
    tied_final = final_counts[final_counts == max_final_votes].index.tolist()
    winner = break_tie(tied_final, tie_breaker_method) if len(tied_final) > 1 else tied_final[0]
    return winner, {"Turno 1": counts_r1, "Resultado Final": final_counts}, len(tied_final) > 1, contingent_votes_series

def run_anti_plurality_n_rejections(df, candidates, tie_breaker_method, num_rejections):
    num_candidates = len(candidates)
    if num_rejections >= num_candidates:
        return "N/A", pd.Series(), True, pd.Series()
    rejection_cols = [f'rank_{i}' for i in range(num_candidates, num_candidates - num_rejections, -1)]
    rejections = df[rejection_cols].values.flatten()
    against_votes = Counter(rejections)
    counts = pd.Series({c: against_votes.get(c, 0) for c in candidates}).sort_values(ascending=True)
    min_votes = counts.min()
    tied = counts[counts == min_votes].index.tolist()
    winner = break_tie(tied, tie_breaker_method) if len(tied) > 1 else tied[0]
    declared_votes = df[rejection_cols].apply(lambda row: tuple(sorted(row)), axis=1)
    return winner, counts, len(tied) > 1, declared_votes

def run_anti_plurality_1_rejection(df, candidates, tie_breaker_method): return run_anti_plurality_n_rejections(df, candidates, tie_breaker_method, 1)
def run_anti_plurality_2_rejections(df, candidates, tie_breaker_method): return run_anti_plurality_n_rejections(df, candidates, tie_breaker_method, 2)
def run_anti_plurality_3_rejections(df, candidates, tie_breaker_method): return run_anti_plurality_n_rejections(df, candidates, tie_breaker_method, 3)

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
    fig.update_layout(title=title, xaxis_title=f'Preferência {c1}', yaxis_title=f'Preferência {c2}', xaxis=dict(range=[-0.1, 1.1]), yaxis=dict(range=[-0.1, 1.1]), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
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
    fig.update_layout(title=title, scene=dict(xaxis_title=f'Pref {c1}', yaxis_title=f'Pref {c2}', zaxis_title=f'Pref {c3}', aspectmode='cube', xaxis=dict(range=[-0.1, 1.1]), yaxis=dict(range=[-0.1, 1.1]), zaxis=dict(range=[-0.1, 1.1])), paper_bgcolor='rgba(0,0,0,0)')
    return fig

def create_parallel_coordinates_plot(df, candidates, title, declared_votes=None):
    fig = go.Figure()
    num_candidates = len(candidates)

    # Define tick values and labels for the main y-axis
    tick_values = [0, 0.25, 0.5, 0.75, 1.0]
    tick_text = ['0.0', '0.25', '0.5', '0.75', '1.0']

    dims = []
    for i, c in enumerate(candidates):
        dim = dict(
            label="",  # Hide the default top label
            values=df[f'pref_{c}'],
            range=[0,1]
        )
        # Only show ticks on the first axis to act as the main Y-axis
        if i == 0:
            dim['tickvals'] = tick_values
            dim['ticktext'] = tick_text
        else:
            # Hide tick labels for other axes
            dim['tickvals'] = []
        dims.append(dim)

    color_values, colorscale = df['Convicção (Desvio Padrão das Prefs.)'], 'Plasma'
    colorbar_title = 'Convicção'

    if declared_votes is not None:
        candidate_map = {name: i for i, name in enumerate(candidates)}
        if isinstance(declared_votes.iloc[0], tuple):
            color_values = declared_votes.apply(lambda x: candidate_map.get(x[0], -1) if x else -1)
        else:
            color_values = declared_votes.map(candidate_map)
        colorscale = 'Rainbow'
        colorbar_title = 'Voto Declarado'

    fig.add_trace(go.Parcoords(
        line=dict(
            color=color_values,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title=colorbar_title,
                x=1.02  # Position colorbar slightly to the right
            )
        ),
        dimensions=dims
    ))

    # --- New Layout and Annotation Logic ---
    annotations = [
        # Y-axis title
        dict(
            x=-0.07,  # Position to the left of the first axis
            y=0.5,
            xref='paper',
            yref='paper',
            text='preferência',
            showarrow=False,
            textangle=-90,
            font=dict(size=14)
        ),
        # X-axis title
        dict(
            x=0.5,
            y=-0.15, # Position below the plot
            xref='paper',
            yref='paper',
            text='candidatos',
            showarrow=False,
            font=dict(size=14)
        )
    ]

    # Add candidate labels at the bottom
    if num_candidates > 0:
        for i, c in enumerate(candidates):
            annotations.append(
                dict(
                    x=i / (num_candidates - 1) if num_candidates > 1 else 0.5,
                    y=-0.07, # Position below the plot, slightly above the main X-axis title
                    xref='paper',
                    yref='paper',
                    text=f'<b>{c}</b>',
                    showarrow=False
                )
            )

    fig.update_layout(
        title=title,
        margin=dict(l=80, r=120, t=80, b=80), # Increase margins for new labels
        annotations=annotations,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig

# --- Definições Globais de Métodos ---
methods_to_analyze = {
    "Pluralidade": run_plurality,
    "Dois Turnos (Top 2)": run_two_round_runoff_top2,
    "Dois Turnos (Top 3)": run_two_round_runoff_top3,
    "Voto Ranqueado (IRV)": run_irv,
    "Voto por Aprovação": run_approval_free,
    "Voto por Aprovação (Livre)": run_approval_free,
    "Voto por Aprovação (1 Fixo)": run_approval_1_fixed,
    "Voto por Aprovação (2 Fixos)": run_approval_2_fixed,
    "Voto por Aprovação (3 Fixos)": run_approval_3_fixed,
    "Voto por Aprovação (4 Fixos)": run_approval_4_fixed,
    "Voto por Aprovação (5 Fixos)": run_approval_5_fixed,
    "Voto por Aprovação (6 Fixos)": run_approval_6_fixed,
    "Voto por Aprovação (7 Fixos)": run_approval_7_fixed,
    "Voto por Aprovação (8 Fixos)": run_approval_8_fixed,
    "Voto por Aprovação (9 Fixos)": run_approval_9_fixed,
    "Voto por Pontuação de 2 Níveis (0-1)": run_score_2_levels,
    "Voto por Pontuação de 3 Níveis (0-2)": run_score_3_levels,
    "Voto por Pontuação de 4 Níveis (0-3)": run_score_4_levels,
    "Voto por Pontuação de 5 Níveis (0-4)": run_score_5_levels,
    "Voto por Pontuação de 6 Níveis (0-5)": run_score_6_levels,
    "Voto por Pontuação de 7 Níveis (0-6)": run_score_7_levels,
    "Voto por Pontuação de 10 Níveis (1-10)": run_score_10_levels,
    "Voto por Pontuação de 2 Níveis Normalizado (0-1)": run_score_2_levels_normalized,
    "Voto por Pontuação de 3 Níveis Normalizado (0-2)": run_score_3_levels_normalized,
    "Voto por Pontuação de 4 Níveis Normalizado (0-3)": run_score_4_levels_normalized,
    "Voto por Pontuação de 5 Níveis Normalizado (0-4)": run_score_5_levels_normalized,
    "Voto por Pontuação de 6 Níveis Normalizado (0-5)": run_score_6_levels_normalized,
    "Voto por Pontuação de 7 Níveis Normalizado (0-6)": run_score_7_levels_normalized,
    "Voto por Pontuação de 10 Níveis Normalizado (1-10)": run_score_10_levels_normalized,
    "Contagem de Borda (Clássica)": run_borda_classic,
    "Contagem de Borda (Dowdall)": run_borda_dowdall,
    "Voto Contingente": run_contingent,
    "Condorcet (2 níveis)": run_condorcet_2_levels,
    "Condorcet (3 níveis)": run_condorcet_3_levels,
    "Condorcet (4 níveis)": run_condorcet_4_levels,
    "Condorcet (5 níveis)": run_condorcet_5_levels,
    "Condorcet (2 níveis normalizado)": run_condorcet_2_levels_normalized,
    "Condorcet (3 níveis normalizado)": run_condorcet_3_levels_normalized,
    "Condorcet (4 níveis normalizado)": run_condorcet_4_levels_normalized,
    "Condorcet (5 níveis normalizado)": run_condorcet_5_levels_normalized,
    "Método de Copeland (Vitórias)": run_copeland_wins,
    "Método de Copeland (Margem)": run_copeland_margin,
    "Anti-Pluralidade (1 em rejeição)": run_anti_plurality_1_rejection,
    "Anti-Pluralidade (2 em rejeição)": run_anti_plurality_2_rejections,
    "Anti-Pluralidade (3 em rejeição)": run_anti_plurality_3_rejections,
}

methods_metadata = {
    "Pluralidade": {
        "Descrição": "Também conhecido como 'First-Past-the-Post', é o sistema mais simples. Cada eleitor escolhe **apenas um** candidato, e o que tiver mais votos vence, mesmo sem maioria absoluta.",
        "Exemplo": "Numa eleição com 100 eleitores para os candidatos A, B e C:\n- A recebe 40 votos.\n- B recebe 35 votos.\n- C recebe 25 votos.\n\n**Resultado:** A vence com 40% dos votos, embora 60% dos eleitores preferissem outro candidato.",
        "Prós": "Simples de entender e apurar.",
        "Contras": "Incentiva o 'voto útil' e pode eleger um candidato que a maioria rejeita.",
        "Competências": "Focado em encontrar um único vencedor de forma rápida. Não gera um ranqueamento completo dos candidatos e ignora as demais preferências dos eleitores."
    },
    "Dois Turnos (Top 2)": {
        "Descrição": "Se nenhum candidato atinge a maioria absoluta (>50%) no primeiro turno, os **dois mais votados** avançam para um segundo turno. Garante que o vencedor tenha o apoio da maioria no confronto final.",
        "Exemplo": "Eleição com 100 eleitores (A, B, C):\n- **1º Turno:** A (40), B (35), C (25).\n- Ninguém tem >50%, então A e B vão para o 2º turno.\n- **2º Turno:** Os eleitores de C agora votam em A ou B.",
        "Prós": "Garante um vencedor com maioria absoluta, dando-lhe mais legitimidade.",
        "Contras": "Pode eliminar candidatos de consenso no 1º turno. Custo e tempo elevados.",
        "Competências": "Focado em encontrar um vencedor com apoio majoritário. Não ranqueia todos os candidatos, apenas os dois finalistas."
    },
    "Dois Turnos (Top 3)": {
        "Descrição": "Uma variação do sistema de dois turnos. Se nenhum candidato atinge a maioria absoluta (>50%) no primeiro turno, os **três mais votados** avançam para um segundo turno. O vencedor do segundo turno é o que tiver mais votos entre os três.",
        "Exemplo": "Eleição com 100 eleitores (A, B, C, D):\n- **1º Turno:** A (35), B (30), C (20), D(15).\n- Ninguém tem >50%, então A, B e C vão para o 2º turno.\n- **2º Turno:** Os eleitores de D votam no seu preferido entre A, B ou C.",
        "Prós": "Dá chance a um candidato de consenso que não ficou no top 2.",
        "Contras": "O vencedor do segundo turno pode não ter a maioria absoluta dos votos originais.",
        "Competências": "Busca um vencedor por pluralidade entre três finalistas. Não gera um ranqueamento completo."
    },
    "Voto Ranqueado (IRV)": {
        "Descrição": "Os eleitores **ranqueiam** os candidatos em ordem de preferência. Se ninguém tem maioria, o candidato com menos votos de 1ª preferência é eliminado. Seus votos são redistribuídos para a próxima preferência de cada eleitor. O processo se repete até um candidato ter maioria.",
        "Exemplo": "100 eleitores (A, B, C):\n- **Rodada 1:** A (40), B (35), C (25). C é eliminado.\n- **Rodada 2:** Os 25 votos de C são transferidos para A ou B com base nas 2ªs preferências.",
        "Prós": "Reduz o 'voto útil' e permite que eleitores votem em quem realmente preferem.",
        "Contras": "Apuração complexa e pode ter resultados não intuitivos.",
        "Competências": "Produz uma ordem de eliminação clara, o que resulta em um ranqueamento completo e sem empates de todos os candidatos (assumindo um método de desempate em cada rodada)."
    },
    "Voto por Aprovação": {
        "Descrição": "Eleitores podem votar em (ou 'aprovar') **quantos candidatos quiserem**. O candidato com o maior número de aprovações vence.",
        "Exemplo": "Numa eleição com 3 eleitores:\n- Eleitor 1 aprova A e B.\n- Eleitor 2 aprova B.\n- Eleitor 3 aprova A, B e C.\n\n**Resultado:** A (2 votos), B (3 votos), C (1 voto). B vence.",
        "Prós": "Simples, expressa apoio amplo e tende a eleger candidatos de menor rejeição.",
        "Contras": "Não permite expressar uma ordem de preferência.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo. No entanto, a probabilidade de empates é alta.",
        "Equivalências": "É funcionalmente muito similar ao 'Voto por Pontuação de 2 Níveis (0-1)'."
    },
    "Voto por Aprovação (Livre)": {
        "Descrição": "Uma variação do Voto por Aprovação onde é explícito que os eleitores podem aprovar **quantos candidatos quiserem**, sem limite.",
        "Exemplo": "Similar ao Voto por Aprovação padrão.",
        "Prós": "Máxima liberdade para o eleitor expressar todo seu apoio.",
        "Contras": "Pode levar a votos estratégicos onde se aprova muitos candidatos para diluir o poder do voto.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo. Alta probabilidade de empates."
    },
    "Voto por Aprovação (1 Fixo)": {
        "Descrição": "Cada eleitor deve votar em **exatamente 1 candidato**. O candidato com o maior número de votos vence.",
        "Exemplo": "Um eleitor vota apenas em A.",
        "Prós": "Extremamente simples e familiar.",
        "Contras": "Não captura nenhuma informação sobre as demais preferências.",
        "Competências": "Não gera um ranqueamento completo, apenas a contagem de votos de primeira preferência.",
        "Equivalências": "Este sistema é funcionalmente idêntico à **Pluralidade**."
    },
    "Voto por Aprovação (2 Fixos)": {
        "Descrição": "Cada eleitor deve votar em **exatamente 2 candidatos**. O candidato com o maior número de votos vence.",
        "Exemplo": "Um eleitor vota em A e B.",
        "Prós": "Permite expressar apoio a mais de um candidato.",
        "Contras": "Força o eleitor a escolher 2, mesmo que só goste de 1 ou de 3.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo. Alta probabilidade de empates."
    },
    "Voto por Aprovação (3 Fixos)": {
        "Descrição": "Cada eleitor deve votar em **exatamente 3 candidatos**. O candidato com o maior número de votos vence.",
        "Exemplo": "Um eleitor vota em A, B e C.",
        "Prós": "Permite expressar apoio a um grupo de candidatos.",
        "Contras": "A obrigatoriedade do número de votos pode ser restritiva.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo. Alta probabilidade de empates."
    },
    "Voto por Aprovação (4 Fixos)": {
        "Descrição": "Cada eleitor deve votar em **exatamente 4 candidatos**. O candidato com o maior número de votos vence.",
        "Exemplo": "Similar aos exemplos anteriores, mas com 4 votos fixos.",
        "Prós": "Permite um apoio ainda mais amplo, útil em eleições com um grande número de candidatos.",
        "Contras": "A obrigatoriedade de votar em 4 pode diluir a preferência real do eleitor. Aumenta a chance de votar em candidatos menos preferidos.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo. Alta probabilidade de empates."
    },
    "Voto por Aprovação (5 Fixos)": {
        "Descrição": "Cada eleitor deve votar em **exatamente 5 candidatos**. O candidato com o maior número de votos vence.",
        "Exemplo": "Similar aos exemplos anteriores, mas com 5 votos fixos.",
        "Prós": "Permite um apoio ainda mais amplo, útil em eleições com um grande número de candidatos.",
        "Contras": "A obrigatoriedade de votar em 5 pode diluir a preferência real do eleitor. Aumenta a chance de votar em candidatos menos preferidos.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo. Alta probabilidade de empates."
    },
    "Voto por Aprovação (6 Fixos)": {
        "Descrição": "Cada eleitor deve votar em **exatamente 6 candidatos**. O candidato com o maior número de votos vence.",
        "Exemplo": "Similar aos exemplos anteriores, mas com 6 votos fixos.",
        "Prós": "Permite um apoio ainda mais amplo, útil em eleições com um grande número de candidatos.",
        "Contras": "A obrigatoriedade de votar em 6 pode diluir a preferência real do eleitor. Aumenta a chance de votar em candidatos menos preferidos.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo. Alta probabilidade de empates."
    },
    "Voto por Aprovação (7 Fixos)": {
        "Descrição": "Cada eleitor deve votar em **exatamente 7 candidatos**. O candidato com o maior número de votos vence.",
        "Exemplo": "Similar aos exemplos anteriores, mas com 7 votos fixos.",
        "Prós": "Permite um apoio ainda mais amplo, útil em eleições com um grande número de candidatos.",
        "Contras": "A obrigatoriedade de votar em 7 pode diluir a preferência real do eleitor. Aumenta a chance de votar em candidatos menos preferidos.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo. Alta probabilidade de empates."
    },
    "Voto por Aprovação (8 Fixos)": {
        "Descrição": "Cada eleitor deve votar em **exatamente 8 candidatos**. O candidato com o maior número de votos vence.",
        "Exemplo": "Similar aos exemplos anteriores, mas com 8 votos fixos.",
        "Prós": "Permite um apoio ainda mais amplo, útil em eleições com um grande número de candidatos.",
        "Contras": "A obrigatoriedade de votar em 8 pode diluir a preferência real do eleitor. Aumenta a chance de votar em candidatos menos preferidos.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo. Alta probabilidade de empates."
    },
    "Voto por Aprovação (9 Fixos)": {
        "Descrição": "Cada eleitor deve votar em **exatamente 9 candidatos**. O candidato com o maior número de votos vence.",
        "Exemplo": "Similar aos exemplos anteriores, mas com 9 votos fixos.",
        "Prós": "Permite um apoio ainda mais amplo, útil em eleições com um grande número de candidatos.",
        "Contras": "A obrigatoriedade de votar em 9 pode diluir a preferência real do eleitor. Aumenta a chance de votar em candidatos menos preferidos.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo. Alta probabilidade de empates."
    },
    "Voto por Pontuação de 2 Níveis (0-1)": {
        "Descrição": "Eleitores dão uma **nota de 0 ou 1** (ex: Ruim/Bom) para cada candidato.",
        "Exemplo": "Eleitor 1: A(1), B(0), C(0). Eleitor 2: A(0), B(1), C(1).\n\n**Resultado:** Soma: A=1, B=1, C=1. Empate.",
        "Prós": "Simples e claro.",
        "Contras": "Pouca granularidade para expressar a intensidade da preferência.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo. Alta probabilidade de empates.",
        "Equivalências": "Este sistema é funcionalmente muito similar ao **Voto por Aprovação**."
    },
    "Voto por Pontuação de 3 Níveis (0-2)": {
        "Descrição": "Eleitores dão uma **nota de 0 a 2** (ex: 0=Ruim, 1=Neutro, 2=Bom) para cada candidato. O candidato com a maior soma total de pontos vence.",
        "Exemplo": "2 eleitores, 3 candidatos (notas de 0 a 2):\n- Eleitor 1: A(2), B(0), C(0).\n- Eleitor 2: A(1), B(2), C(1).\n\n**Resultado:** Soma A=3, B=2, C=1. A vence.",
        "Prós": "Simples, permite expressar alguma intensidade de preferência sem ser excessivamente granular. Tende a eleger candidatos de consenso.",
        "Contras": "Menos expressivo que escalas maiores. Ainda vulnerável a estratégias de 'rebaixamento'.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo e detalhado. A probabilidade de empates é menor que no Voto por Aprovação."
    },
    "Voto por Pontuação de 4 Níveis (0-3)": {
        "Descrição": "Eleitores dão uma **nota de 0 a 3** (ex: 0=Péssimo, 1=Ruim, 2=Bom, 3=Excelente) para cada candidato. O candidato com a maior soma total de pontos vence.",
        "Exemplo": "2 eleitores, 3 candidatos (notas de 0 a 3):\n- Eleitor 1: A(3), B(0), C(1).\n- Eleitor 2: A(2), B(3), C(2).\n\n**Resultado:** Soma A=5, B=3, C=3. A vence.",
        "Prós": "Oferece mais granularidade que 3 níveis, permitindo uma expressão mais matizada da preferência. Promove candidatos de consenso.",
        "Contras": "Pode ser um pouco mais complexo para o eleitor decidir entre 4 níveis. Vulnerável a estratégias de 'rebaixamento'.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo e detalhado. A probabilidade de empates é menor que no Voto por Aprovação."
    },
    "Voto por Pontuação de 5 Níveis (0-4)": {
        "Descrição": "Eleitores dão uma **nota de 0 a 4** (ex: 0=Péssimo, 1=Ruim, 2=Neutro, 3=Bom, 4=Excelente) para cada candidato. O candidato com a maior soma total de pontos vence.",
        "Exemplo": "2 eleitores, 3 candidatos (notas de 0 a 4):\n- Eleitor 1: A(4), B(0), C(1).\n- Eleitor 2: A(3), B(4), C(2).\n\n**Resultado:** Soma A=7, B=4, C=3. A vence.",
        "Prós": "Escala intuitiva com um ponto neutro, permitindo uma boa expressão da intensidade da preferência. Amplamente utilizada em sistemas de avaliação.",
        "Contras": "Ainda vulnerável a estratégias de 'rebaixamento' e 'inflação'.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo e detalhado. A probabilidade de empates é menor que no Voto por Aprovação."
    },
    "Voto por Pontuação de 6 Níveis (0-5)": {
        "Descrição": "Eleitores dão uma **nota de 0 a 5** (ex: 0=Péssimo, 1=Muito Ruim, 2=Ruim, 3=Bom, 4=Muito Bom, 5=Excelente) para cada candidato. O candidato com a maior soma total de pontos vence.",
        "Exemplo": "2 eleitores, 3 candidatos (notas de 0 a 5):\n- Eleitor 1: A(5), B(0), C(1).\n- Eleitor 2: A(4), B(5), C(2).\n\n**Resultado:** Soma A=9, B=5, C=3. A vence.",
        "Prós": "Oferece uma boa granularidade para expressar nuances de preferência. Promove candidatos de alto consenso.",
        "Contras": "Pode ser um pouco mais complexo para o eleitor. Vulnerável a estratégias de 'rebaixamento' e 'inflação'.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo e detalhado. A probabilidade de empates é menor que no Voto por Aprovação."
    },
    "Voto por Pontuação de 7 Níveis (0-6)": {
        "Descrição": "Eleitores dão uma **nota de 0 a 6** (ex: 0=Péssimo, 1=Muito Ruim, 2=Ruim, 3=Neutro, 4=Bom, 5=Muito Bom, 6=Excelente) para cada candidato. O candidato com a maior soma total de pontos vence.",
        "Exemplo": "2 eleitores, 3 candidatos (notas de 0 a 6):\n- Eleitor 1: A(6), B(0), C(1).\n- Eleitor 2: A(5), B(6), C(2).\n\n**Resultado:** Soma A=11, B=6, C=3. A vence.",
        "Prós": "Oferece uma granularidade ainda maior com um ponto neutro, permitindo uma expressão muito detalhada da preferência. Ideal para capturar consenso.",
        "Contras": "Pode ser mais difícil para o eleitor diferenciar entre tantos níveis. Vulnerável a estratégias de 'rebaixamento' e 'inflação'.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo e detalhado. A probabilidade de empates é menor que no Voto por Aprovação."
    },
    "Voto por Pontuação de 10 Níveis (1-10)": {
        "Descrição": "Eleitores dão uma **nota de 1 a 10** para cada candidato. O candidato com a maior nota média (ou soma total) vence. Permite expressar a intensidade da preferência.",
        "Exemplo": "2 eleitores, 3 candidatos (notas de 1 a 10):\n- Eleitor 1: A(10), B(2), C(1).\n- Eleitor 2: A(4), B(10), C(8).\n\n**Resultado:** Soma A=14, B=12, C=9. A vence.",
        "Prós": "Captura a intensidade da preferência e promove candidatos de alto consenso. Simples de entender para o eleitor.",
        "Contras": "Vulnerável a estratégias de 'rebaixamento' (dar nota mínima a concorrentes fortes) e 'inflação' (dar nota máxima ao favorito e mínima aos outros).",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo e detalhado. A probabilidade de empates é menor que no Voto por Aprovação."
    },
    "Voto por Pontuação de 2 Níveis Normalizado (0-1)": {
        "Descrição": "Eleitores dão uma **nota de 0 ou 1** para cada candidato. As notas de cada eleitor são então **normalizadas** para que a soma total seja 1, garantindo que cada eleitor tenha o mesmo peso total. O candidato com a maior soma de notas normalizadas vence.",
        "Exemplo": "Eleitor 1: A(1), B(0), C(0). Notas normalizadas: A(1), B(0), C(0).\nEleitor 2: A(0), B(1), C(1). Notas normalizadas: A(0), B(0.5), C(0.5).\n\n**Resultado:** Soma: A=1, B=0.5, C=0.5. A vence.",
        "Prós": "Reduz o impacto de estratégias de 'rebaixamento' e 'inflação' de notas.",
        "Contras": "Pode ser menos intuitivo para o eleitor.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo e detalhado."
    },
    "Voto por Pontuação de 3 Níveis Normalizado (0-2)": {
        "Descrição": "Eleitores dão uma **nota de 0 a 2** para cada candidato. As notas de cada eleitor são então **normalizadas** para que a soma total seja 1, garantindo que cada eleitor tenha o mesmo peso total. O candidato com a maior soma de notas normalizadas vence.",
        "Exemplo": "Eleitor 1: A(2), B(0), C(0). Notas normalizadas: A(1), B(0), C(0).\nEleitor 2: A(1), B(2), C(1). Notas normalizadas: A(0.25), B(0.5), C(0.25).\n\n**Resultado:** Soma: A=1.25, B=0.5, C=0.25. A vence.",
        "Prós": "Reduz o impacto de estratégias de 'rebaixamento' e 'inflação' de notas.",
        "Contras": "Pode ser menos intuitivo para o eleitor.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo e detalhado."
    },
    "Voto por Pontuação de 4 Níveis Normalizado (0-3)": {
        "Descrição": "Eleitores dão uma **nota de 0 a 3** para cada candidato. As notas de cada eleitor são então **normalizadas** para que a soma total seja 1, garantindo que cada eleitor tenha o mesmo peso total. O candidato com a maior soma de notas normalizadas vence.",
        "Exemplo": "Eleitor 1: A(3), B(0), C(1). Notas normalizadas: A(0.75), B(0), C(0.25).\nEleitor 2: A(2), B(3), C(2). Notas normalizadas: A(0.28), B(0.43), C(0.28).\n\n**Resultado:** Soma: A=1.03, B=0.43, C=0.53. A vence.",
        "Prós": "Reduz o impacto de estratégias de 'rebaixamento' e 'inflação' de notas.",
        "Contras": "Pode ser menos intuitivo para o eleitor.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo e detalhado."
    },
    "Voto por Pontuação de 5 Níveis Normalizado (0-4)": {
        "Descrição": "Eleitores dão uma **nota de 0 a 4** para cada candidato. As notas de cada eleitor são então **normalizadas** para que a soma total seja 1, garantindo que cada eleitor tenha o mesmo peso total. O candidato com a maior soma de notas normalizadas vence.",
        "Exemplo": "Eleitor 1: A(4), B(0), C(1). Notas normalizadas: A(0.8), B(0), C(0.2).\nEleitor 2: A(3), B(4), C(2). Notas normalizadas: A(0.33), B(0.44), C(0.22).\n\n**Resultado:** Soma: A=1.13, B=0.44, C=0.42. A vence.",
        "Prós": "Reduz o impacto de estratégias de 'rebaixamento' e 'inflação' de notas.",
        "Contras": "Pode ser menos intuitivo para o eleitor.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo e detalhado."
    },
    "Voto por Pontuação de 6 Níveis Normalizado (0-5)": {
        "Descrição": "Eleitores dão uma **nota de 0 a 5** para cada candidato. As notas de cada eleitor são então **normalizadas** para que a soma total seja 1, garantindo que cada eleitor tenha o mesmo peso total. O candidato com a maior soma de notas normalizadas vence.",
        "Exemplo": "Eleitor 1: A(5), B(0), C(1). Notas normalizadas: A(0.83), B(0), C(0.17).\nEleitor 2: A(4), B(5), C(2). Notas normalizadas: A(0.36), B(0.45), C(0.18).\n\n**Resultado:** Soma: A=1.19, B=0.45, C=0.35. A vence.",
        "Prós": "Reduz o impacto de estratégias de 'rebaixamento' e 'inflação' de notas.",
        "Contras": "Pode ser menos intuitivo para o eleitor.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo e detalhado."
    },
    "Voto por Pontuação de 7 Níveis Normalizado (0-6)": {
        "Descrição": "Eleitores dão uma **nota de 0 a 6** para cada candidato. As notas de cada eleitor são então **normalizadas** para que a soma total seja 1, garantindo que cada eleitor tenha o mesmo peso total. O candidato com a maior soma de notas normalizadas vence.",
        "Exemplo": "Eleitor 1: A(6), B(0), C(1). Notas normalizadas: A(0.86), B(0), C(0.14).\nEleitor 2: A(5), B(6), C(2). Notas normalizadas: A(0.38), B(0.46), C(0.15).\n\n**Resultado:** Soma: A=1.24, B=0.46, C=0.29. A vence.",
        "Prós": "Reduz o impacto de estratégias de 'rebaixamento' e 'inflação' de notas.",
        "Contras": "Pode ser menos intuitivo para o eleitor.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo e detalhado."
    },
    "Voto por Pontuação de 10 Níveis Normalizado (1-10)": {
        "Descrição": "Eleitores dão uma **nota de 1 a 10** para cada candidato. As notas de cada eleitor são então **normalizadas** para que a soma total seja 1 (com a nota mínima tratada como 0), garantindo que cada eleitor tenha o mesmo peso total. O candidato com a maior soma de notas normalizadas vence.",
        "Exemplo": "Eleitor 1: A(10), B(1). Notas normalizadas: A(1), B(0).\nEleitor 2: A(6), B(6). Notas normalizadas: A(0.5), B(0.5).\n\n**Resultado:** Soma: A=1.5, B=0.5. A vence.",
        "Prós": "Reduz o impacto de estratégias de 'rebaixamento' e 'inflação' de notas, pois o peso total do voto é constante.",
        "Contras": "Pode ser menos intuitivo para o eleitor entender como sua preferência é traduzida em peso final.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo e detalhado. A normalização pode alterar significativamente os resultados em comparação com a versão não normalizada."
    },
    "Contagem de Borda (Clássica)": {
        "Descrição": "Eleitores ranqueiam os candidatos. Cada posição no ranking vale pontos (ex: 1º lugar = N-1 pontos, 2º = N-2, etc.). O candidato com mais pontos vence.",
        "Exemplo": "3 candidatos (A,B,C). 1º=2pts, 2º=1pt, 3º=0pts. Eleitor vota (A > B > C).\n\n**Resultado:** Pontos: A=2, B=1, C=0.",
        "Prós": "Elege candidatos de consenso, que podem não ser o favorito da maioria, mas são amplamente aceitáveis.",
        "Contras": "Vulnerável à clonagem de candidatos e ao 'rebaixamento' estratégico.",
        "Competências": "Sempre gera um ranqueamento completo de todos os candidatos, pois cada um recebe uma pontuação. Empates são possíveis, mas menos frequentes que na Aprovação."
    },
    "Contagem de Borda (Dowdall)": {
        "Descrição": "Uma variação da Contagem de Borda onde os pontos são atribuídos de forma harmônica. O 1º lugar recebe 1 ponto, o 2º lugar 1/2 ponto, o 3º 1/3, e assim por diante.",
        "Exemplo": "Eleitor vota (A > B > C).\n\n**Resultado:** Pontos: A=1, B=0.5, C=0.33.",
        "Prós": "Dá um peso muito maior para as primeiras posições, valorizando a primeira escolha do eleitor.",
        "Contras": "Menos comum e pode ser menos intuitivo. Ainda vulnerável a estratégias.",
        "Competências": "Sempre gera um ranqueamento completo e com pontuações bem distintas, diminuindo a chance de empates em comparação com a Borda Clássica."
    },
    "Voto Contingente": {
        "Descrição": "Uma versão simplificada do IRV. Se ninguém tem maioria, **todos os candidatos são eliminados, exceto os dois primeiros**. Os votos dos eleitores dos candidatos eliminados são então transferidos para um dos dois finalistas.",
        "Exemplo": "A(35), B(30), C(20), D(15). A e B avançam. Votos de C e D são transferidos para A ou B.",
        "Prós": "Mais simples que o IRV, mas ainda garante um vencedor com maioria.",
        "Contras": "Pode eliminar um 'candidato de consenso' no primeiro turno.",
        "Competências": "Focado em encontrar um vencedor majoritário. Não gera um ranqueamento completo dos candidatos."
    },
    "Condorcet (2 níveis)": {
        "Descrição": "Em cada duelo 1x1, o eleitor expressa uma preferência binária (A ou B). O placar final de um candidato é a soma de suas margens de vitória (ou derrota) em todos os duelos.",
        "Exemplo": "A vs B: 60 eleitores preferem A, 40 preferem B. Margem para A é +20.\nA vs C: 30 preferem A, 70 preferem C. Margem para A é -40.\n\n**Resultado:** Placar de A = 20 - 40 = -20.",
        "Prós": "Simples de entender e considera a força da vitória em cada par.",
        "Contras": "Pode ser menos expressivo que métodos com mais níveis.",
        "Competências": "Gera um score que reflete a performance geral de um candidato em duelos."
    },
    "Condorcet (3 níveis)": {
        "Descrição": "Em cada duelo 1x1, o eleitor pode expressar preferência por A, por B, ou um voto neutro. A preferência é baseada na diferença de nota entre os candidatos (threshold de 33%). O placar final é a soma das margens de vitória.",
        "Exemplo": "A vs B: 50 preferem A, 20 preferem B, 30 são neutros. Margem para A é +30.\nA vs C: 20 preferem A, 60 preferem C, 20 são neutros. Margem para A é -40.\n\n**Resultado:** Placar de A = 30 - 40 = -10.",
        "Prós": "Permite que eleitores expressem indiferença, reduzindo o ruído de pequenas preferências.",
        "Contras": "A definição de 'neutro' pode ser arbitrária.",
        "Competências": "Gera um score baseado em preferências mais fortes."
    },
    "Condorcet (4 níveis)": {
        "Descrição": "Em cada duelo 1x1, o eleitor expressa 2 níveis de preferência para cada candidato (ex: 'Mais para A', 'Muito mais para A'). A preferência é baseada na diferença de nota (thresholds de 0% e 50%). O placar final é a soma dos pontos.",
        "Exemplo": "A vs B: A ganha 2 pontos para cada 'Muito mais para A' e 1 ponto para cada 'Mais para A'. O mesmo para B. A margem é a diferença dos pontos totais.",
        "Prós": "Captura a intensidade da preferência em confrontos diretos.",
        "Contras": "Complexidade aumentada.",
        "Competências": "Gera um score baseado em uma avaliação granular das preferências."
    },
    "Condorcet (5 níveis)": {
        "Descrição": "Em cada duelo 1x1, o eleitor expressa 2 níveis de preferência para cada candidato ou um voto neutro (thresholds de 20% e 60%). O placar final é a soma dos pontos.",
        "Exemplo": "A vs B: A ganha 2 pontos para cada 'Muito mais para A', 1 para 'Mais para A', 0 para neutro. O mesmo para B. A margem é a diferença dos pontos totais.",
        "Prós": "Oferece alta granularidade para expressar a intensidade da preferência.",
        "Contras": "Muito complexo para uma eleição real.",
        "Competências": "Gera um score baseado em uma avaliação muito detalhada das preferências."
    },
    "Condorcet (2 níveis normalizado)": {
        "Descrição": "Similar ao Condorcet de 2 níveis, mas os 'pontos' de cada eleitor em todos os duelos são normalizados para somar 1, garantindo que cada eleitor tenha o mesmo peso total.",
        "Exemplo": "As pontuações de um eleitor em todos os pares (A vs B, A vs C, etc.) são calculadas e depois divididas pela soma total (absoluta) dessas pontuações.",
        "Prós": "Reduz o impacto de votos estratégicos onde um eleitor expressa preferências extremas.",
        "Contras": "A normalização pode ser complexa e pouco intuitiva.",
        "Competências": "Busca um vencedor Condorcet mais robusto a estratégias."
    },
    "Condorcet (3 níveis normalizado)": {
        "Descrição": "Similar ao Condorcet com 3 níveis, mas os 'pontos' de preferência de cada eleitor em todos os confrontos 1x1 são normalizados para somar 1.",
        "Exemplo": "As pontuações de preferência de um eleitor em todos os pares são calculadas e depois divididas pela soma total (absoluta) dessas pontuações.",
        "Prós": "Reduz o impacto de votos estratégicos onde um eleitor expressa preferências extremas em apenas alguns pares.",
        "Contras": "A normalização pode ser complexa e pouco intuitiva.",
        "Competências": "Busca um vencedor Condorcet mais robusto a estratégias de manipulação de margens."
    },
    "Condorcet (4 níveis normalizado)": {
        "Descrição": "Similar ao Condorcet com 4 níveis, mas os 'pontos' de preferência de cada eleitor em todos os confrontos 1x1 são normalizados para somar 1.",
        "Exemplo": "As pontuações de preferência de um eleitor em todos os pares são calculadas e depois divididas pela soma total (absoluta) dessas pontuações.",
        "Prós": "Reduz o impacto de votos estratégicos onde um eleitor expressa preferências extremas.",
        "Contras": "A normalização pode ser complexa e pouco intuitiva.",
        "Competências": "Busca um vencedor Condorcet mais robusto a estratégias de manipulação de margens."
    },
    "Condorcet (5 níveis normalizado)": {
        "Descrição": "Similar ao Condorcet com 5 níveis, mas os 'pontos' de preferência de cada eleitor em todos os confrontos 1x1 são normalizados para somar 1.",
        "Exemplo": "As pontuações de preferência de um eleitor em todos os pares são calculadas e depois divididas pela soma total (absoluta) dessas pontuações.",
        "Prós": "Reduz o impacto de votos estratégicos onde um eleitor expressa preferências extremas.",
        "Contras": "A normalização pode ser complexa e pouco intuitiva.",
        "Competências": "Busca um vencedor Condorcet mais robusto a estratégias de manipulação de margens."
    },
    "Método de Copeland (Vitórias)": {
        "Descrição": "Uma forma de encontrar um vencedor usando o critério Condorcet. O vencedor é o candidato que **vence o maior número de confrontos diretos** (um-contra-um).",
        "Exemplo": "A vs B (A vence)\nA vs C (A vence)\nB vs C (B vence)\n\n**Resultado:** Placar: A (2 vitórias), B (1), C (0). A vence.",
        "Prós": "Sempre produz um resultado e é baseado no 'justo' critério Condorcet.",
        "Contras": "Pode resultar em empates e a apuração é complexa.",
        "Competências": "Gera um score (número de vitórias) para cada candidato, permitindo um ranqueamento completo. Empates são comuns."
    },
    "Método de Copeland (Margem)": {
        "Descrição": "Uma variação do método de Copeland onde o placar de cada candidato é a **soma das margens de vitória** em seus confrontos diretos (votos a favor - votos contra).",
        "Exemplo": "A vs B (10-5), A vs C (12-3).\n\n**Resultado:** Margem de A = (10-5) + (12-3) = 14.",
        "Prós": "Leva em conta a 'força' da vitória, não apenas o número de vitórias.",
        "Contras": "Pode ser menos intuitivo que a contagem de vitórias simples.",
        "Competências": "Gera um score para cada candidato, permitindo um ranqueamento completo e com menos probabilidade de empates que a versão por vitórias."
    },
    "Anti-Pluralidade (1 em rejeição)": {
        "Descrição": "Também conhecido como 'Voto de Rejeição'. Cada eleitor vota no candidato que **menos** deseja. O candidato com o menor número de votos 'contra' é o vencedor.",
        "Exemplo": "100 eleitores rejeitam um candidato.\nA: 10 votos 'contra'\nB: 30 'contra'\nC: 60 'contra'\n\n**Resultado:** A vence.",
        "Prós": "Simples e eficaz para evitar a eleição de um candidato amplamente impopular.",
        "Contras": "Não expressa preferência positiva.",
        "Competências": "Gera um score de rejeição para cada candidato, permitindo um ranqueamento completo (do menos ao mais rejeitado). Empates são possíveis."
    },
    "Anti-Pluralidade (2 em rejeição)": {
        "Descrição": "Cada eleitor vota nos **dois** candidatos que menos deseja. O(s) candidato(s) com o menor número total de votos 'contra' vence(m).",
        "Exemplo": "Eleitor rejeita A e B.",
        "Prós": "Permite uma expressão de rejeição mais ampla.",
        "Contras": "Pode ser difícil para o eleitor escolher múltiplos candidatos para rejeitar.",
        "Competências": "Gera um score de rejeição para cada candidato, permitindo um ranqueamento completo. Empates são possíveis."
    },
    "Anti-Pluralidade (3 em rejeição)": {
        "Descrição": "Cada eleitor vota nos **três** candidatos que menos deseja. O(s) candidato(s) com o menor número total de votos 'contra' vence(m).",
        "Exemplo": "Eleitor rejeita A, B e C.",
        "Prós": "Útil em cenários com muitos candidatos para filtrar os mais indesejados.",
        "Contras": "Aumenta a complexidade para o eleitor.",
        "Competências": "Gera um score de rejeição para cada candidato, permitindo um ranqueamento completo. Empates são possíveis."
    }
}


# --- Interface Principal ---
st.title("📊 Ferramenta de Análise Estatística de Sistemas de Votação")

with st.sidebar:
    st.header("Configurações Gerais")
    num_agents = st.number_input("Número de Agentes por Simulação", min_value=2, max_value=10000, value=100, step=1)
    num_candidates = st.slider("Número de Candidatos", 2, 25, 3, 1)
    strategic_prob = st.slider("Probabilidade de Voto Estratégico", 0.0, 1.0, 0.25, 0.05, help="Simula o 'voto útil' de compromisso.")
    tie_breaker_method = st.selectbox("Em caso de Empate/Paradoxo", ["Anulação da Votação", "Aleatório", "Ordem Alfabética"])
    
    st.markdown("---")
    st.header("Análise Estatística")
    num_simulations = st.slider("Número de Simulações", 10, 10000, 100, 10)
    run_analysis = st.button("Executar Análise Estatística", type="primary")

# --- Lógica Principal ---
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
        
        condorcet_winner, _, _, _ = run_condorcet(agents_df, candidates, tie_breaker_method)
        if condorcet_winner == "Paradoxo": condorcet_winner = None
        
        utilitarian_winner = run_score_10_levels(agents_df, candidates, tie_breaker_method)[0]
        if utilitarian_winner == "Anulada": utilitarian_winner = None

        majority_counts = agents_df['rank_1'].value_counts()
        majority_winner = majority_counts.index[0] if not majority_counts.empty and majority_counts.iloc[0] > num_agents / 2 else None

        plurality_winner = run_plurality(agents_df, candidates, tie_breaker_method)[0]
        if plurality_winner == "Anulada": plurality_winner = None

        for name, method_func in methods_to_analyze.items():
            # Adicionado tratamento de erro para funções que podem falhar com poucos candidatos
            try:
                honest_winner, _, _, _ = method_func(agents_df, candidates, tie_breaker_method)
                strategic_winner, _, _, _ = method_func(strategic_df, candidates, tie_breaker_method)
            except (IndexError, KeyError):
                honest_winner, strategic_winner = "N/A", "N/A"

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
    desc_df = pd.DataFrame.from_dict(methods_metadata, orient='index')
    desc_df.index.name = "Método"
    desc_df = desc_df.reset_index()

    for method in methods_to_analyze.keys():
        method_df = summary_df[summary_df['method'] == method]
        
        num_total_sims = len(method_df)
        if num_total_sims == 0: continue

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

    # Filter methods based on num_candidates
    filtered_methods_df = final_df.copy()
    filtered_desc_df = desc_df.copy()
    methods_to_remove = []

    # Filter fixed approval methods
    if num_candidates < 10:
        for i in range(num_candidates, 10):
            methods_to_remove.append(f"Voto por Aprovação ({i} Fixos)")

    # Filter anti-plurality methods
    if num_candidates < 3:
        methods_to_remove.append("Anti-Pluralidade (2 em rejeição)")
    if num_candidates < 4:
        methods_to_remove.append("Anti-Pluralidade (3 em rejeição)")

    # Filter runoff methods
    if num_candidates < 4:
        methods_to_remove.append("Dois Turnos (Top 3)")

    filtered_methods_df = filtered_methods_df[~filtered_methods_df['Método'].isin(methods_to_remove)].reset_index(drop=True)
    filtered_desc_df = filtered_desc_df[~filtered_desc_df['Método'].isin(methods_to_remove)].reset_index(drop=True)

    st.session_state.analysis_results_df = filtered_methods_df
    st.session_state.analysis_desc_df = filtered_desc_df
    st.session_state.last_example_df = last_run_df
    st.session_state.last_example_candidates = last_run_candidates

if st.session_state.get('analysis_results_df') is not None:
    st.info("A tabela abaixo classifica os sistemas por um 'Score Final'. **Clique em uma linha** para ver a descrição detalhada e a análise do sistema selecionado.")
    
    results_df = st.session_state.analysis_results_df
    desc_df = st.session_state.analysis_desc_df
    
    if results_df is None or desc_df is None:
        st.warning("Os resultados da análise não estão disponíveis.")
        st.stop()

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
    if st.session_state.system_selector['selection']['rows']:
        selected_index = st.session_state.system_selector['selection']['rows'][0]
        selected_method_name = results_df.loc[int(selected_index), 'Método']

    st.markdown("---")
    
    selected_method_info = desc_df[desc_df['Método'] == selected_method_name].iloc[0]
    
    with st.expander(f"Como funciona: {selected_method_name}", expanded=True):
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.subheader("Descrição")
            st.markdown(selected_method_info['Descrição'])
            st.subheader("Exemplo Prático")
            st.markdown(selected_method_info['Exemplo'])
        with col2:
            st.subheader("Prós")
            st.success(selected_method_info['Prós'])
            st.subheader("Contras")
            st.error(selected_method_info['Contras'])
            if pd.notna(selected_method_info.get("Competências")):
                st.subheader("Competências")
                st.info(selected_method_info['Competências'])
            if pd.notna(selected_method_info.get("Equivalências")):
                st.subheader("Equivalências")
                st.warning(selected_method_info['Equivalências'])

    st.markdown("---")
    st.header(f"Análise da Última Simulação para '{selected_method_name}'")
    st.warning("Os dados abaixo são da **última** simulação executada, aplicando o sistema selecionado acima.")

    example_voting_func = methods_to_analyze[selected_method_name]
    
    example_df = st.session_state.last_example_df
    example_candidates = st.session_state.last_example_candidates

    if example_df is None or example_candidates is None:
        st.warning("Os dados da última simulação não estão disponíveis.")
        st.stop()

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
        # Add a widget for the user to choose the chart type
        chart_type = st.selectbox(
            "Escolha o tipo de gráfico",
            ("Coordenadas Paralelas", "Gráfico 2D", "Gráfico 3D"),
            help="Selecione a forma de visualização dos dados dos agentes."
        )

        # Display the selected chart
        if chart_type == "Gráfico 2D":
            if len(example_candidates) >= 2:
                default_candidates = example_candidates[:2] if example_candidates else []
                selected_candidates = st.multiselect("Selecione 2 candidatos", example_candidates, default=default_candidates)
                if len(selected_candidates) == 2:
                    fig = create_2d_plot(example_df, selected_candidates, plot_title)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Por favor, selecione exatamente 2 candidatos.")
            else:
                st.warning("O Gráfico 2D requer pelo menos 2 candidatos.")
        elif chart_type == "Gráfico 3D":
            if len(example_candidates) >= 3:
                default_candidates = example_candidates[:3] if example_candidates else []
                selected_candidates = st.multiselect("Selecione 3 candidatos", example_candidates, default=default_candidates)
                if len(selected_candidates) == 3:
                    fig = create_3d_plot(example_df, selected_candidates, plot_title)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Por favor, selecione exatamente 3 candidatos.")
            else:
                st.warning("O Gráfico 3D requer pelo menos 3 candidatos.")
        else:  # Coordenadas Paralelas
            if len(example_candidates) > 0:
                fig = create_parallel_coordinates_plot(example_df, example_candidates, plot_title)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("O Gráfico de Coordenadas Paralelas requer pelo menos 1 candidato.")
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
        elif selected_method_name in ["Dois Turnos (Top 2)", "Dois Turnos (Top 3)", "Voto Contingente"]:
            if "Resultado Final" in results:
                st.text("Preferências de 1º Turno:")
                st.dataframe(results["Turno 1"], use_container_width=True)
                st.text("Resultado Final:")
                st.dataframe(results["Resultado Final"], use_container_width=True)
            elif "Turno 2" in results:
                st.text("Turno 1:")
                st.dataframe(results["Turno 1"], use_container_width=True)
                st.text(f"Finalistas: {results['Finalistas']}")
                st.text("Turno 2:")
                st.dataframe(results["Turno 2"], use_container_width=True)
            else:
                st.text("Turno 1 (Vencedor por maioria):")
                st.dataframe(results["Turno 1"], use_container_width=True)
        elif isinstance(results, pd.Series):
            results_df = results.reset_index()
            results_df.columns = ['Candidato', 'Pontuação']
            st.dataframe(results_df, use_container_width=True)
        else:
            st.dataframe(results, use_container_width=True)
else:
    st.info("Ajuste os parâmetros na barra lateral e clique em 'Executar Análise Estatística' para começar.")