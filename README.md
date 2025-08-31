# 🗳️ Simulador de Sistemas de Votação

Uma ferramenta interativa para simular e analisar estatisticamente o desempenho de diferentes sistemas de votação. Este aplicativo permite aos usuários comparar visualmente como vários métodos (Pluralidade, Voto Ranqueado, Aprovação, etc.) se comportam sob diferentes cenários.

![Captura de tela da aplicação](https://raw.githubusercontent.com/Edamas/simulador-sistemas-votacao/main/Captura%20de%20tela%202025-08-29%20021540.jpg)

## ✨ Funcionalidades

*   **Simulação Estatística:** Execute centenas ou milhares de eleições para avaliar a performance de cada sistema.
*   **Métricas de Avaliação:** Os sistemas são classificados por um Score Final com base em:
    *   **Justiça (Critério Condorcet):** A capacidade de eleger o candidato que venceria todos os outros em confrontos diretos.
    *   **Satisfação (Vencedor Utilitário):** A capacidade de eleger o candidato com a maior preferência média.
    *   **Respeito à Maioria:** A capacidade de eleger um candidato que tem mais de 50% dos votos de primeira preferência.
    *   **Resistência à Estratégia:** A resiliência do sistema ao "voto útil".
*   **Análise Detalhada:** Explore visualizações interativas e os dados brutos da última simulação para entender o comportamento dos eleitores.
*   **Configuração Flexível:** Ajuste o número de eleitores, candidatos e a probabilidade de voto estratégico.

## ⚙️ Tecnologias Utilizadas

*   **Python**
*   **Streamlit:** Para a interface web interativa.
*   **Pandas & NumPy:** Para manipulação e análise de dados.
*   **Plotly:** Para as visualizações de dados.

## 🚀 Como Executar

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/Edamas/simulador-sistemas-votacao.git
    cd simulador-sistemas-votacao
    ```

2.  **Instale as dependências:**
    É recomendado criar um ambiente virtual.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Execute o aplicativo Streamlit:**
    ```bash
    streamlit run app.py
    ```

## 🗳️ Sistemas Analisados

| Método | Descrição Resumida |
| --- | --- |
| **Pluralidade** | O candidato com mais votos de primeira preferência vence. |
| **Dois Turnos (Top 2 / Top 3)** | Os 2 ou 3 mais votados vão para um segundo turno se ninguém atingir a maioria. |
| **Voto Ranqueado (IRV)** | Candidatos menos populares são eliminados em rodadas e seus votos redistribuídos. |
| **Voto por Aprovação** | Eleitores podem 'aprovar' quantos candidatos quiserem (versões Livre e Fixa). |
| **Voto por Pontuação** | Eleitores dão notas aos candidatos e o que tiver a maior média vence (múltiplas escalas, normalizadas ou não). |
| **Contagem de Borda (Clássica / Dowdall)** | Candidatos recebem pontos com base na sua posição no ranking de cada eleitor. |
| **Voto Contingente** | Versão simplificada do IRV, com apenas uma rodada de redistribuição. |
| **Condorcet** | Métodos que buscam o candidato que vence todos os outros em confrontos diretos (múltiplas variações). |
| **Método de Copeland (Vitórias / Margem)** | Vencedor baseado no número ou na margem das vitórias em confrontos diretos. |
| **Anti-Pluralidade (Rejeição)** | Eleitores votam no(s) candidato(s) que menos desejam; o menos rejeitado vence. |

## Descrição Detalhada dos Sistemas

### Pluralidade

**Descrição:** Também conhecido como 'First-Past-the-Post', é o sistema mais simples. Cada eleitor escolhe **apenas um** candidato, e o que tiver mais votos vence, mesmo sem maioria absoluta.

**Prós:** Simples de entender e apurar.

**Contras:** Incentiva o 'voto útil' e pode eleger um candidato que a maioria rejeita.

### Dois Turnos (Runoff)

**Descrição:** Se nenhum candidato atinge a maioria absoluta (>50%) no primeiro turno, os mais votados avançam para um segundo turno.
*   **Variações no Simulador:**
    *   **Top 2:** Os dois mais votados avançam. Garante que o vencedor tenha o apoio da maioria no confronto final.
    *   **Top 3:** Os três mais votados avançam. Dá chance a um candidato de consenso que não ficou no top 2, mas o vencedor do 2º turno pode não ter maioria absoluta.

**Prós:** Garante um vencedor com maioria absoluta, dando-lhe mais legitimidade.

**Contras:** Mais caro e demorado. Candidatos de consenso, mas menos populares, podem ser eliminados no 1º turno.

### Voto Ranqueado (IRV)

**Descrição:** Os eleitores **ranqueiam** os candidatos em ordem de preferência. Se ninguém tem maioria, o candidato com menos votos de 1ª preferência é eliminado. Seus votos são redistribuídos para a próxima preferência de cada eleitor. O processo se repete até um candidato ter maioria.

**Prós:** Reduz o 'voto útil' e permite que eleitores votem em quem realmente preferem. Elege vencedores de maior consenso.

**Contras:** Apuração complexa e pode ter resultados não intuitivos.

### Voto por Aprovação

**Descrição:** Eleitores podem votar em (ou 'aprovar') **quantos candidatos quiserem**. O candidato com o maior número de aprovações vence. É como uma eleição de múltipla escolha.
*   **Variações no Simulador:**
    *   **Voto por Aprovação (Livre):** Implementação padrão onde os eleitores aprovam qualquer candidato acima de um limiar de preferência.
    *   **Voto por Aprovação (X Fixo):** Variações onde o eleitor **deve** votar em exatamente X candidatos (de 1 a 9). Isso força uma escolha mais estruturada.

**Prós:** Simples, expressa apoio amplo e tende a eleger candidatos de menor rejeição.

**Contras:** Não permite expressar uma ordem de preferência. Um eleitor que aprova seu favorito e um candidato 'aceitável' dá o mesmo peso a ambos.

### Voto por Pontuação

**Descrição:** Eleitores dão uma **nota** para cada candidato (ex: 0 a 10). O candidato com a maior nota média (ou soma total) vence. Permite expressar a intensidade da preferência.
*   **Variações no Simulador:**
    *   O simulador inclui escalas de 2, 3, 4, 5, 6, 7 e 10 níveis para analisar como a granularidade da pontuação afeta o resultado.
    *   **Normalizado:** As notas de cada eleitor são normalizadas para que a soma total seja 1, garantindo que cada eleitor tenha o mesmo peso total. Isso reduz o impacto de estratégias de 'rebaixamento' e 'inflação' de notas.

**Prós:** Captura a intensidade da preferência e promove candidatos de alto consenso.

**Contras:** Vulnerável a estratégias de 'rebaixamento' (dar nota mínima a concorrentes fortes) e 'inflação' (dar nota máxima ao favorito).

### Contagem de Borda

**Descrição:** Eleitores ranqueiam os candidatos. Cada posição no ranking vale pontos.
*   **Variações no Simulador:**
    *   **Clássica:** Sistema de pontos linear (N-1, N-2, ..., 0).
    *   **Dowdall:** Sistema de pontos harmônico (1, 1/2, 1/3, ...), que dá mais peso às primeiras posições.

**Prós:** Elege candidatos de consenso, que podem não ser o favorito da maioria, mas são amplamente aceitáveis.

**Contras:** Vulnerável à clonagem de candidatos e ao 'rebaixamento' estratégico.

### Voto Contingente

**Descrição:** Uma versão simplificada do IRV. Se ninguém tem maioria, **todos os candidatos são eliminados, exceto os dois primeiros**. Os votos dos eleitores dos candidatos eliminados são então transferidos para um dos dois finalistas, conforme a preferência.

**Prós:** Mais simples que o IRV, mas ainda garante um vencedor com maioria.

**Contras:** Pode eliminar um 'candidato de consenso' no primeiro turno.

### Condorcet

**Descrição:** Um método teórico que define o vencedor 'ideal'. O vencedor Condorcet é o candidato que, em **confrontos diretos (um-contra-um)**, venceria todos os outros candidatos.
*   **Variações no Simulador:**
    *   **Níveis (2 a 5):** Em cada duelo 1x1, o eleitor pode expressar diferentes níveis de preferência, que se traduzem em pontos. O placar final de um candidato é a soma de suas margens de vitória (ou derrota) em todos os duelos.
    *   **Normalizado:** Os 'pontos' de cada eleitor em todos os duelos são normalizados para somar 1, garantindo que cada eleitor tenha o mesmo peso total.

**Prós:** Considerado o critério mais 'justo' de uma eleição.

**Contras:** Pode não haver um vencedor (Paradoxo de Condorcet), tornando-o mais um critério de avaliação do que um método prático.

### Método de Copeland

**Descrição:** Uma forma de encontrar um vencedor usando o critério Condorcet.
*   **Variações no Simulador:**
    *   **Vitórias:** O vencedor é o que vence o maior número de confrontos diretos.
    *   **Margem:** O vencedor é o que tem a maior margem de vitória acumulada (votos a favor - votos contra) em todos os confrontos.

**Prós:** Sempre produz um resultado e é baseado no critério Condorcet.

**Contras:** Pode resultar em empates e a apuração é complexa.

### Anti-Pluralidade

**Descrição:** Também conhecido como 'Voto de Rejeição'. Cada eleitor vota no candidato que **menos** deseja. O candidato com o menor número de votos 'contra' é o vencedor.
*   **Variações no Simulador:**
    *   **1, 2 ou 3 em rejeição:** O eleitor deve rejeitar 1, 2 ou 3 candidatos.

**Prós:** Simples e eficaz para evitar a eleição de um candidato amplamente impopular.

**Contras:** Não expressa preferência positiva e pode eleger um candidato que poucos realmente apoiam.
