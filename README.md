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

*   Pluralidade (Primeiro a Passar do Posto)
*   Dois Turnos (Runoff)
*   Voto Ranqueado (IRV)
*   Voto por Aprovação
*   Voto por Pontuação
*   Contagem de Borda
*   Voto Contingente
*   Método de Condorcet
*   Método de Copeland
*   Anti-Pluralidade
