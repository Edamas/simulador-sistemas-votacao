# 🗳️ Simulador de Sistemas de Votação

Uma ferramenta interativa para simular e analisar estatisticamente o desempenho de diferentes sistemas de votação. Este aplicativo permite aos usuários comparar visualmente como vários métodos (Pluralidade, Voto Ranqueado, Aprovação, etc.) se comportam sob diferentes cenários.

![Captura de tela da aplicação](https://raw.githubusercontent.com/Edamas/simulador-sistemas-votacao/main/Captura%20de%20tela%202025-08-29%20021540.jpg)
*(Lembre-se de substituir SEU-USUARIO e SEU-REPOSITORIO no link da imagem acima depois de enviar a imagem para o seu repositório)*

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
    git clone https://github.com/SEU-USUARIO/simulador-sistemas-votacao.git
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
| **Pluralidade** | Também conhecido como 'First-Past-the-Post', é o sistema mais simples.|
| **Dois Turnos (Runoff)** | Se nenhum candidato atinge a maioria absoluta (>50%) no primeiro turno, os **dois mais votados** avançam para um segundo turno.|
| **Voto Ranqueado (IRV)** | Os eleitores **ranqueiam** os candidatos em ordem de preferência.|
| **Voto por Aprovação** | Eleitores podem votar em (ou 'aprovar') **quantos candidatos quiserem**.|
| **Voto por Pontuação** | Eleitores dão uma **nota** (ex: 0 a 10) para cada candidato.|
| **Contagem de Borda** | Eleitores ranqueiam os candidatos.|
| **Voto Contingente** | Uma versão simplificada do IRV.|
| **Condorcet** | Um método teórico que define o vencedor 'ideal'.|
| **Método de Copeland** | Uma forma de encontrar um vencedor usando o critério Condorcet.|
| **Anti-Pluralidade** | Também conhecido como 'Voto de Rejeição'.|

## Descrição Detalhada dos Sistemas
### Pluralidade

**Descrição:** Também conhecido como 'First-Past-the-Post', é o sistema mais simples. Cada eleitor escolhe **apenas um** candidato, e o que tiver mais votos vence, mesmo sem maioria absoluta.

**Exemplo:**

Numa eleição com 100 eleitores para os candidatos A, B e C:
- A recebe 40 votos.
- B recebe 35 votos.
- C recebe 25 votos.
**Resultado:** A vence com 40% dos votos, embora 60% dos eleitores preferissem outro candidato.

**Prós:** Simples de entender e apurar.

**Contras:** Incentiva o 'voto útil' e pode eleger um candidato que a maioria rejeita.

### Dois Turnos (Runoff)

**Descrição:** Se nenhum candidato atinge a maioria absoluta (>50%) no primeiro turno, os **dois mais votados** avançam para um segundo turno. Isso garante que o vencedor tenha o apoio da maioria.

**Exemplo:**

Eleição com 100 eleitores (A, B, C):
- **1º Turno:** A (40), B (35), C (25).
- Ninguém tem >50%, então A e B vão para o 2º turno.
- **2º Turno:** Os eleitores de C agora votam em A ou B. Se a maioria deles preferir B, o resultado pode ser B (35+15=50) vs A (40+10=50), levando a um empate, ou um deles vencer.

**Prós:** Garante um vencedor com maioria absoluta, dando-lhe mais legitimidade.

**Contras:** Mais caro e demorado. Candidatos de consenso, mas menos populares, podem ser eliminados no 1º turno.

### Voto Ranqueado (IRV)

**Descrição:** Os eleitores **ranqueiam** os candidatos em ordem de preferência. Se ninguém tem maioria, o candidato com menos votos de 1ª preferência é eliminado. Seus votos são redistribuídos para a próxima preferência de cada eleitor. O processo se repete até um candidato ter maioria.

**Exemplo:**

100 eleitores (A, B, C):
- **Rodada 1:** A (40), B (35), C (25). C é eliminado.
- **Rodada 2:** Os 25 votos de C são transferidos. Se 15 deles tinham B como 2ª opção e 10 tinham A:
  - A: 40 + 10 = 50
  - B: 35 + 15 = 50
**Resultado:** Empate entre A e B. O IRV busca um vencedor de consenso.

**Prós:** Reduz o 'voto útil' e permite que eleitores votem em quem realmente preferem. Elege vencedores de maior consenso.

**Contras:** Apuração complexa e pode ter resultados não intuitivos (ex: um candidato pode vencer mesmo que outro fosse preferido contra ele em um confronto direto).

### Voto por Aprovação

**Descrição:** Eleitores podem votar em (ou 'aprovar') **quantos candidatos quiserem**. O candidato com o maior número de aprovações vence. É como uma eleição de múltipla escolha.

**Exemplo:**

Numa eleição com 3 eleitores:
- Eleitor 1 aprova A e B.
- Eleitor 2 aprova B.
- Eleitor 3 aprova A, B e C.
**Resultado:** A (2 votos), B (3 votos), C (1 voto). B vence.

**Prós:** Simples, expressa apoio amplo e tende a eleger candidatos de menor rejeição.

**Contras:** Não permite expressar uma ordem de preferência. Um eleitor que aprova seu favorito e um candidato 'aceitável' dá o mesmo peso a ambos.

### Voto por Pontuação

**Descrição:** Eleitores dão uma **nota** (ex: 0 a 10) para cada candidato. O candidato com a maior nota média (ou soma total) vence. Permite expressar a intensidade da preferência.

**Exemplo:**

2 eleitores, 3 candidatos (notas de 0 a 5):
- Eleitor 1: A(5), B(1), C(0).
- Eleitor 2: A(2), B(5), C(4).
**Resultado:** Soma A=7, B=6, C=4. A vence.

**Prós:** Captura a intensidade da preferência e promove candidatos de alto consenso.

**Contras:** Vulnerável a estratégias de 'rebaixamento', onde eleitores dão nota mínima a concorrentes fortes para ajudar seu favorito.

### Contagem de Borda

**Descrição:** Eleitores ranqueiam os candidatos. Cada posição no ranking vale pontos (ex: 1º lugar = N-1 pontos, 2º = N-2, etc., onde N é o nº de candidatos). O candidato com mais pontos vence.

**Exemplo:**

3 candidatos (A,B,C). 1º=2pts, 2º=1pt, 3º=0pts. 5 eleitores:
- 3 eleitores: (A > B > C) -> A: 3*2=6, B: 3*1=3
- 2 eleitores: (B > C > A) -> B: 2*2=4, C: 2*1=2
**Resultado:** A=6, B=7, C=2. B vence, pois foi bem ranqueado por todos.

**Prós:** Elege candidatos de consenso, que podem não ser o favorito da maioria, mas são amplamente aceitáveis.

**Contras:** Vulnerável à clonagem de candidatos (candidatos semelhantes podem dividir os pontos de um oponente forte) e ao 'rebaixamento' estratégico.

### Voto Contingente

**Descrição:** Uma versão simplificada do IRV. Se ninguém tem maioria, **todos os candidatos são eliminados, exceto os dois primeiros**. Os votos dos eleitores dos candidatos eliminados são então transferidos para um dos dois finalistas, conforme a preferência.

**Exemplo:**

100 eleitores (A, B, C, D):
- **1º Turno:** A(35), B(30), C(20), D(15). Ninguém tem >50%.
- **Contingência:** A e B avançam. Os votos de C e D são transferidos para A ou B. Se a maioria dos eleitores de C e D preferem B a A, B pode vencer.

**Prós:** Mais simples que o IRV, mas ainda garante um vencedor com maioria.

**Contras:** Pode eliminar um 'candidato de consenso' no primeiro turno (ex: um candidato que seria a 2ª opção de todos).

### Condorcet

**Descrição:** Um método teórico que define o vencedor 'ideal'. O vencedor Condorcet é o candidato que, em **confrontos diretos (um-contra-um)**, venceria todos os outros candidatos.

**Exemplo:**

Candidatos A, B, C. A maioria prefere A a B, A a C, e B a C.
- A vs B -> A vence.
- A vs C -> A vence.
- B vs C -> B vence.
**Resultado:** A é o vencedor Condorcet.

**Prós:** Considerado o critério mais 'justo' de uma eleição.

**Contras:** Pode não haver um vencedor (Paradoxo de Condorcet, ex: A>B, B>C, C>A), tornando-o mais um critério de avaliação do que um método prático.

### Método de Copeland

**Descrição:** Uma forma de encontrar um vencedor usando o critério Condorcet. O vencedor é o candidato que **vence o maior número de confrontos diretos** (um-contra-um).

**Exemplo:**

A vs B (A vence), A vs C (A vence), B vs C (B vence).
- **Placar:** A (2 vitórias), B (1 vitória), C (0 vitórias).
**Resultado:** A vence.

**Prós:** Sempre produz um resultado e é baseado no 'justo' critério Condorcet.

**Contras:** Pode resultar em empates e a apuração é complexa.

### Anti-Pluralidade

**Descrição:** Também conhecido como 'Voto de Rejeição'. Cada eleitor vota no candidato que **menos** deseja. O candidato com o menor número de votos 'contra' é o vencedor.

**Exemplo:**

100 eleitores votam para rejeitar A, B ou C:
- A recebe 10 votos 'contra'.
- B recebe 30 votos 'contra'.
- C recebe 60 votos 'contra'.
**Resultado:** A vence, por ser o menos rejeitado.

**Prós:** Simples e eficaz para evitar a eleição de um candidato amplamente impopular.

**Contras:** Não expressa preferência positiva e pode eleger um candidato que poucos realmente apoiam.