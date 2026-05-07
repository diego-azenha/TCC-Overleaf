#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gera estrutura LaTeX do TCC - FactorVAE no Mercado Brasileiro."""
import os, re

BASE = r"m:\Python Projects\TCC_VAE_OVERLEAF"

def w(rel, content):
    path = os.path.join(BASE, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content.lstrip('\n'))
    print(f"  OK: {rel}")

# ============================================================
# CAPÍTULO 1 — INTRODUÇÃO
# ============================================================
CAP1 = r"""
\chapter{Introdução}
\label{cap:introducao}

A identificação de fatores sistemáticos de risco constitui uma das tarefas centrais das finanças modernas. Modelos de fatores cumprem três funções econômicas interligadas: decompor o risco de portfólios em componentes sistemáticos e idiossincráticos, estimar matrizes de covariância em grandes \textit{cross-sections} a partir de estruturas parcimoniosamente parametrizadas, e formar expectativas condicionais de retorno como combinações lineares de exposições a fatores comuns. A sequência canônica dessa literatura parte do \textit{Capital Asset Pricing Model} de \textcite{sharpe1964} e da teoria de precificação por arbitragem de \textcite{ross1976}, avança para os modelos de fatores construídos a partir de características observáveis de \textcite{fama1993} e \textcite{fama2015}, e constitui a espinha dorsal da gestão quantitativa de portfólios até hoje.

A limitação central desses modelos reside na especificação ex-ante dos fatores: o pesquisador precisa decidir, antes de ver os dados, quais características são portadoras de prêmio de risco sistemático. Essa dependência de hipóteses prévias, combinada com a restrição da linearidade na estimação das exposições, motivou uma virada metodológica em direção a abordagens que aprendem fatores diretamente dos dados. \textcite{gu2021} demonstraram que autoencoders condicionais superam sistematicamente modelos lineares e redes neurais sem estrutura fatorial na previsão do \textit{cross-section} de retornos norte-americanos. A limitação dessas abordagens é que produzem estimativas pontuais de retorno, operando sob baixa razão sinal-ruído sem mecanismo explícito para tratar essa fonte de incerteza. A extensão probabilística surge como tentativa de endereçar o problema de forma estrutural: em vez de estimar o retorno esperado pontualmente, o modelo aprende a distribuição condicional completa dos retornos, capturando heterogeneidade na incerteza tanto idiossincrática quanto sistemática.

O \textit{FactorVAE}, proposto por \textcite{duan2022}, representa essa extensão probabilística dos modelos fatoriais neurais. A arquitetura combina uma estrutura de fatores latentes com inferência variacional: durante o treinamento, um codificador extrai uma distribuição \textit{posterior} sobre os fatores a partir dos retornos contemporâneos; um preditor aprende a distribuição \textit{prior} correspondente a partir exclusivamente de informação histórica; e a função objetivo força a aproximação entre as duas distribuições, transferindo ao preditor o conteúdo informativo capturado pelo codificador. Na inferência fora da amostra, apenas o preditor e o decodificador participam — o codificador é descartado por depender dos retornos realizados, indisponíveis no momento da previsão. O modelo foi avaliado no mercado A-share chinês, com universo superior a três mil ativos e período de teste de dois anos. O comportamento do \textit{FactorVAE} em mercados com algumas centenas de ativos, custos de transação não-triviais e período de teste mais longo não foi documentado. A pergunta que orienta este trabalho é direta: o \textit{FactorVAE} replicado fielmente no mercado brasileiro mantém vantagem de \textit{skill} cross-sectional sobre baselines neurais, e essa vantagem sobrevive à tradução em portfólio sob custos de transação realistas?

O restante do trabalho está organizado da seguinte forma. O Capítulo~\ref{cap:revisao} revisa a evolução dos modelos de fatores, apresenta os fundamentos de inferência variacional necessários para compreender a função objetivo do modelo e descreve os procedimentos de avaliação que serão aplicados. O Capítulo~\ref{cap:metodologia} especifica operacionalmente a arquitetura, os dados e os procedimentos de comparação. O Capítulo~\ref{cap:resultados} apresenta e interpreta os resultados. O Capítulo~\ref{cap:conclusao} sintetiza os achados, discute limitações e propõe extensões.
"""

# ============================================================
# CAPÍTULO 2 — REVISÃO DA LITERATURA
# ============================================================
CAP2 = r"""
\chapter{Revisão da Literatura}
\label{cap:revisao}

A revisão é deliberadamente enxuta: o capítulo entrega apenas o que o leitor precisa para acompanhar a metodologia. O lugar do \textit{FactorVAE} na evolução da modelagem fatorial, o vocabulário de inferência variacional necessário para entender sua função objetivo, a apresentação do modelo em si e os procedimentos de avaliação que serão aplicados aos resultados.

\section{Modelos de Fatores: dos Observáveis aos Latentes}
\label{sec:modelos-fatores}

Modelos de fatores expressam o retorno de um ativo como combinação de exposições a fatores de risco comuns mais um componente idiossincrático. O \textit{Capital Asset Pricing Model} de \textcite{sharpe1964} introduz o caso de fator único — o excesso de retorno do portfólio de mercado — com exposições estimadas por regressão linear de séries temporais. A teoria de precificação por arbitragem de \textcite{ross1976} generaliza essa estrutura para múltiplos fatores sem exigir que o portfólio de mercado seja o único fator relevante. Em ambos os casos, a estrutura geral do modelo é:

\begin{equation}
    r_{i,t} = \alpha_i + \sum_{k=1}^{K} \beta_{i,k}\, f_{k,t} + \varepsilon_{i,t}
    \label{eq:modelo-fatorial}
\end{equation}

\noindent em que $r_{i,t}$ é o retorno do ativo $i$ no período $t$, $f_{k,t}$ são os fatores comuns, $\beta_{i,k}$ são as exposições do ativo $i$ ao fator $k$, $\alpha_i$ é o retorno esperado não explicado pelos fatores, e $\varepsilon_{i,t}$ é o componente idiossincrático, assumido não correlacionado entre ativos e com os fatores.

A segunda geração de modelos abandona a derivação de equilíbrio e constrói fatores diretamente a partir de características observáveis dos ativos. \textcite{fama1993} propõem portfólios formados com base em tamanho e índice book-to-market como proxies para fatores de risco não capturados pelo mercado. \textcite{fama2015} estendem o modelo para cinco fatores, incorporando rentabilidade e investimento. \textcite{carhart1997} adiciona um fator de momento. A lógica é a mesma: hipóteses ex-ante sobre quais características geram prêmios sistemáticos, estimação por regressão linear de painel, exposições constantes no tempo.

Três limitações estruturais motivam a busca por alternativas. Primeiro, a dependência da especificação ex-ante: o modelo é tão bom quanto as hipóteses do pesquisador sobre quais características são portadoras de prêmio. Segundo, a restrição linear, que impede a captura de interações não-lineares entre características e retornos. Terceiro, a proliferação de centenas de fatores documentados sem critério de seleção sistemático — o chamado \textit{factor zoo} —, que torna arbitrária a escolha do conjunto de características.

Essas limitações motivaram uma linha de pesquisa que aprende a representação fatorial diretamente dos dados. \textcite{gu2021} propõem o \textit{Conditional Autoencoder}: os fatores latentes são extraídos por um autoencoder, e as exposições são funções não-lineares aprendidas das características dos ativos. O modelo supera sistematicamente modelos lineares e redes neurais sem estrutura fatorial na previsão do \textit{cross-section} de retornos norte-americanos. A limitação é que produz estimativas pontuais: o modelo entrega o retorno esperado condicional sem modelar a distribuição condicional completa. O \textit{FactorVAE} estende essa linha ao tratar os fatores latentes como variáveis aleatórias e modelar explicitamente a incerteza sobre eles, preparando o terreno para a subseção~\ref{sec:factorvae}.

\section{Variational Autoencoders}
\label{sec:vae}

O problema central dos \textit{Variational Autoencoders} (VAEs) é a estimação de um modelo generativo com variáveis latentes quando a verossimilhança marginal não admite forma fechada. Dado um vetor de observações $x$ e variáveis latentes $z$, a verossimilhança marginal $p(x) = \int p(x|z)\,p(z)\,\mathrm{d}z$ não tem solução analítica quando o decodificador $p(x|z)$ é parametrizado por uma rede neural não-linear, tornando inviável a estimação direta por máxima verossimilhança.

A inferência variacional resolve esse problema introduzindo uma distribuição aproximada $q(z|x)$ — o codificador — e maximizando um limite inferior da log-verossimilhança, o ELBO (\textit{Evidence Lower Bound}):

\begin{equation}
    \log p(x) \;\geq\; \mathbb{E}_{q(z|x)}\!\left[\log p(x|z)\right] - \mathrm{KL}\!\left(q(z|x) \,\|\, p(z)\right)
    \label{eq:elbo}
\end{equation}

\noindent O primeiro termo mede a qualidade da reconstrução de $x$ a partir das variáveis latentes $z$ amostradas da posterior; o segundo penaliza o quanto a posterior $q(z|x)$ se afasta da prior $p(z)$. O truque de reparametrização de \textcite{kingma2014} permite que o gradiente flua através da operação de amostragem ao expressar $z = \mu + \sigma \odot \epsilon$ com $\epsilon \sim \mathcal{N}(0, I)$, tornando o ELBO diferenciável com respeito aos parâmetros de ambas as redes.

No caso padrão, a prior é fixa em $\mathcal{N}(0, I)$ e o termo KL tem forma fechada analítica. Quando a prior é aprendida — parametrizada por uma rede separada —, o termo KL compara duas distribuições gaussianas com parâmetros distintos, e o comportamento numérico do treinamento muda: os gradientes da prior dependem agora de uma rede que precisa ser otimizada conjuntamente com o codificador. Esse ponto é diretamente relevante para o \textit{FactorVAE}: o codificador produz uma posterior e o preditor produz uma prior, ambos aprendidos conjuntamente, e a função objetivo força a aproximação entre essas duas distribuições ao longo do treinamento.

\section{O Modelo FactorVAE}
\label{sec:factorvae}

O \textit{FactorVAE} \parencite{duan2022} preserva a decomposição econômica fundamental dos modelos fatoriais: o retorno de cada ativo é a soma de um componente idiossincrático e um componente sistemático, este último escrito como combinação linear de fatores latentes. A inovação não está na decomposição em si, e sim no fato de que todos os componentes — o retorno idiossincrático esperado $\alpha$, a matriz de exposições $\beta$ e os próprios fatores $z$ — são saídas de redes neurais que processam exclusivamente informação histórica disponível antes da data de previsão. Isso elimina a dependência de hipóteses ex-ante sobre quais características são relevantes e permite que o modelo capture relações não-lineares entre características e retornos.

\begin{figure}[htbp]
    \centering
    \fbox{\parbox{0.82\textwidth}{\centering\vspace{2.5cm}
    {\small \textit{PLACEHOLDER} --- Diagrama geral da arquitetura do \textit{FactorVAE}.}\\[0.4cm]
    {\small Quatro blocos: \textbf{Extrator de Características}, \textbf{Codificador},
    \textbf{Preditor} e \textbf{Decodificador}.}\\[0.2cm]
    {\small Fluxo de dados entre blocos, com indicação visual de quais participam}\\
    {\small apenas do treinamento (Codificador + retornos $y_t$) e quais participam}\\
    {\small da inferência (Extrator, Preditor, Decodificador).}\\[0.4cm]
    {\small Adaptada de \textcite{duan2022}.}
    \vspace{2.5cm}}}
    \caption{Diagrama geral da arquitetura do \textit{FactorVAE}.}
    \label{fig:factorvae-geral}
\end{figure}

O traço arquitetural que distingue o \textit{FactorVAE} de um VAE convencional é o esquema professor-aluno entre codificador e preditor. Durante o treinamento, o codificador recebe os retornos contemporâneos $y_t$ como sinal adicional e produz uma distribuição posterior $q(z|x_t, y_t)$ sobre os fatores; o preditor recebe apenas a informação histórica $x_t$ e produz uma distribuição prior $p(z|x_t)$. A função objetivo força o preditor a aprender os fatores de risco relevantes ao aproximar sua prior da posterior informada pelo oráculo. Na inferência fora da amostra, os retornos contemporâneos não estão disponíveis, e o codificador é descartado: apenas o preditor e o decodificador participam da previsão. Esse esquema é o mecanismo central do modelo; sem ele, o preditor não teria sinal suficiente para aprender os fatores latentes a partir exclusivamente de informação histórica.

Uma propriedade adicional é a composição analítica gaussiana no decodificador. Sob a hipótese de que tanto o componente idiossincrático quanto os fatores têm distribuições gaussianas condicionalmente independentes, a distribuição preditiva dos retornos é também gaussiana com parâmetros em forma fechada, sem necessidade de amostragem Monte Carlo. Isso entrega não apenas o retorno esperado por ativo, mas também uma estimativa de incerteza por ativo $\sigma_y^{(i)}$, aproveitável na construção de estratégias de portfólio ajustadas ao risco.

\section{Avaliação de Modelos Preditivos em Finanças}
\label{sec:avaliacao-lit}

A avaliação de modelos preditivos em finanças enfrenta um problema de dois estágios: medir a qualidade do sinal preditivo e medir a qualidade da estratégia resultante. O Rank IC é definido como a correlação de Spearman entre os retornos previstos e os retornos realizados no \textit{cross-section} em cada data $t$; o Rank ICIR é a razão entre a média e o desvio-padrão do Rank IC ao longo do tempo, capturando a consistência do sinal. Um Rank IC positivo indica que o modelo posiciona, em média, os ativos de melhor desempenho nas primeiras posições do ranking; o Rank ICIR penaliza a instabilidade temporal desse sinal.

A tradução do sinal em portfólio é feita pela estratégia TopK-Drop \parencite{yang2020}: a cada data, selecionam-se os $k$ ativos com maior retorno previsto, rebalanceando a carteira com no máximo $n$ substituições por período para controlar o turnover. As métricas de portfólio avaliadas são: retorno anualizado, retorno em excesso sobre o benchmark, volatilidade anualizada, Índice de Sharpe (retorno em excesso por unidade de risco), Information Ratio (excesso anualizado dividido pelo desvio do excesso), Índice de Calmar (retorno anualizado dividido pelo drawdown máximo), drawdown máximo, hit rate (fração de períodos com retorno positivo) e turnover médio diário.

A disciplina amostral é condição necessária para a validade dos resultados. Viés de sobrevivência surge quando ativos que deslistaram durante o período de análise são excluídos retroativamente do universo; a construção point-in-time elimina esse viés ao reconstruir a composição do universo disponível em cada data histórica sem informação prospectiva. Os subperíodos de validação e teste devem permanecer intocados durante a estimação do modelo: qualquer normalização, seleção de hiperparâmetros ou pré-processamento ancorado no período de teste constitui contaminação amostral. Esses são os pontos que serão cobrados nas decisões metodológicas do Capítulo~\ref{cap:metodologia}.
"""

# ============================================================
# CAPÍTULO 3 — METODOLOGIA
# ============================================================
CAP3 = r"""
\chapter{Metodologia}
\label{cap:metodologia}

Este capítulo transita da estrutura conceitual apresentada na revisão para a especificação operacional do modelo, com ênfase no que precisa ser definido para que o trabalho seja replicável. A descrição cobre a formulação matemática do problema, a arquitetura do \textit{FactorVAE} em seus quatro blocos, o procedimento de treinamento, os dados utilizados, os modelos de comparação e os procedimentos de avaliação.

\section{Formulação do Problema}
\label{sec:formulacao}

O modelo busca a distribuição condicional dos retornos cross-sectionais no período $t{+}1$ dado o conjunto de informações disponível até o final do período $t$. A entrada do modelo é um tensor de características históricas $x_t \in \mathbb{R}^{N_t \times T \times C}$, em que $N_t$ é o número de ativos disponíveis na data $t$ (variável ao longo do tempo, aproximadamente 300--500 no período de teste), $T = 20$ é o comprimento da janela temporal em dias úteis e $C = 20$ é o número de características por ativo por período. A saída é uma distribuição preditiva sobre o vetor de retornos $r_{t+1} \in \mathbb{R}^{N_t}$.

A decomposição fatorial que o modelo preserva é:

\begin{equation}
    r_{t+1} = \alpha(x_t) + \beta(x_t)\, z_{t+1} + \varepsilon_{t+1}
    \label{eq:decomp-operacional}
\end{equation}

\noindent em que $\alpha(x_t) \in \mathbb{R}^{N_t}$ é o vetor de componentes idiossincráticos esperados, $\beta(x_t) \in \mathbb{R}^{N_t \times K}$ é a matriz de exposições, $z_{t+1} \in \mathbb{R}^K$ é o vetor de fatores latentes e $\varepsilon_{t+1}$ é o resíduo idiossincrático. Todas as três quantidades são funções aprendidas de $x_t$. Os parâmetros de dimensão adotados são: $K = 16$ fatores latentes, $M = 64$ portfólios construídos internamente pelo codificador, e dimensão oculta $H$ a ser especificada na Tabela~\ref{tab:hiperparametros}. As escolhas seguem \textcite{duan2022} para garantir comparabilidade direta com o estudo original.

\section{Estrutura do Modelo}
\label{sec:estrutura-modelo}

A arquitetura é organizada em quatro blocos sequenciais: Extrator de Características, Codificador de Fatores, Preditor de Fatores e Decodificador de Fatores. Os dois primeiros são usados apenas durante o treinamento; os dois últimos participam também da inferência fora da amostra.

\begin{figure}[htbp]
    \centering
    \fbox{\parbox{0.82\textwidth}{\centering\vspace{2.5cm}
    {\small \textit{PLACEHOLDER} --- Diagrama detalhado da arquitetura.}\\[0.4cm]
    {\small Versão granular mostrando as camadas internas de cada bloco,}\\
    {\small as dimensões dos tensores trocados e os caminhos de gradiente}\\
    {\small durante o treinamento.}\\[0.4cm]
    {\small Elaboração própria.}
    \vspace{2.5cm}}}
    \caption{Diagrama detalhado da arquitetura do \textit{FactorVAE}.}
    \label{fig:factorvae-detalhado}
\end{figure}

\subsection{Extrator de Características}
\label{subsec:extrator}

O Extrator de Características processa a janela histórica de cada ativo de forma independente, com pesos compartilhados entre ativos. Para cada ativo $i$, uma projeção linear seguida de LeakyReLU mapeia o vetor de características de dimensão $C$ para a dimensão oculta $H$. Uma GRU \parencite{cho2014} processa a sequência projetada ao longo dos $T = 20$ dias. O último estado oculto $e^{(i)} \in \mathbb{R}^H$ é a representação por ativo, capturando o padrão temporal relevante de cada série de características. O compartilhamento de pesos entre ativos é a propriedade que permite ao modelo generalizar para ativos não vistos durante o treinamento, conforme avaliado na Seção~\ref{sec:avaliacao-robustez}.

\subsection{Codificador de Fatores}
\label{subsec:codificador}

O Codificador de Fatores recebe as representações por ativo $\{e^{(i)}\}_{i=1}^{N}$ e os retornos contemporâneos $y_t \in \mathbb{R}^N$ e produz a distribuição posterior $q(z|x_t, y_t) = \mathcal{N}(\mu_{\text{post}}, \text{diag}(\sigma_{\text{post}}^2))$.

A \textit{PortfolioLayer} parametriza $M = 64$ portfólios cujos pesos dependem das representações. Para o $j$-ésimo portfólio, o peso do ativo $i$ é:

\begin{equation}
    a^{(i,j)} = \frac{\exp\!\left((W_p\, e^{(i)} + b_p)^{(j)}\right)}{\displaystyle\sum_{i'=1}^{N} \exp\!\left((W_p\, e^{(i')} + b_p)^{(j)}\right)}
    \label{eq:pesos-portfolio}
\end{equation}

\noindent em que $W_p$ e $b_p$ são parâmetros aprendidos. Os pesos derivam exclusivamente das representações históricas $e^{(i)}$, não dos retornos; os retornos contemporâneos entram apenas como o sinal a ser ponderado. O retorno do $j$-ésimo portfólio é:

\begin{equation}
    y_p^{(j)} = \sum_{i=1}^{N} y^{(i)}\, a^{(i,j)}
    \label{eq:retorno-portfolio}
\end{equation}

A \textit{MappingLayer} projeta o vetor de retornos de portfólio $y_p \in \mathbb{R}^M$ nos parâmetros da posterior via duas camadas lineares independentes, com Softplus garantindo $\sigma_{\text{post}} > 0$. O codificador funciona como oráculo: sua posterior é informada pelos retornos realizados, impossíveis de observar na inferência. A validade desse oráculo — isto é, a confirmação de que a posterior captura sinal genuíno e não apenas ruído — é verificada pelo diagnóstico de Posterior IC reportado no Apêndice~\ref{ap:diagnosticos}.

\subsection{Preditor de Fatores}
\label{subsec:preditor}

O Preditor de Fatores recebe as representações $\{e^{(i)}\}$ e produz a distribuição prior $p(z|x_t) = \mathcal{N}(\mu_{\text{prior}}, \text{diag}(\sigma_{\text{prior}}^2))$ sem acesso aos retornos contemporâneos. Ele é composto por $K = 16$ cabeças de atenção independentes, cada uma dedicada a estimar os parâmetros de um fator.

Cada cabeça mantém uma query aprendida $q^{(k)} \in \mathbb{R}^H$ e calcula pesos de atenção via similaridade cosseno com gate ReLU:

\begin{equation}
    a_{\text{att}}^{(i,k)} = \frac{\max\!\left(0,\; \cos(q^{(k)},\, e^{(i)})\right)}{\displaystyle\sum_{i'} \max\!\left(0,\; \cos(q^{(k)},\, e^{(i')})\right)}
    \label{eq:atencao-relu}
\end{equation}

\noindent O gate ReLU descarta ativos com similaridade negativa e normaliza sobre os ativos restantes, diferindo de uma atenção \textit{softmax} convencional que distribui peso não-nulo a todos os ativos. O agregado de mercado para o fator $k$ é $c^{(k)} = \sum_i a_{\text{att}}^{(i,k)}\, e^{(i)}$. Uma \textit{DistributionNetwork} compartilhada entre as $K$ cabeças projeta $c^{(k)}$ nos parâmetros $(\mu_{\text{prior}}^{(k)}, \sigma_{\text{prior}}^{(k)})$.

\subsection{Decodificador de Fatores}
\label{subsec:decodificador}

O Decodificador de Fatores recebe as representações $\{e^{(i)}\}$ e os fatores amostrados $z$ e produz a distribuição preditiva sobre os retornos. Uma \textit{AlphaLayer} projeta $e^{(i)}$ nos parâmetros $(\mu_\alpha^{(i)}, \sigma_\alpha^{(i)})$ do componente idiossincrático via duas camadas lineares com Softplus. Uma \textit{BetaLayer} projeta $e^{(i)}$ na linha $i$ da matriz de exposições $\beta^{(i)} \in \mathbb{R}^K$ via projeção linear.

Sob independência condicional entre o componente idiossincrático e os fatores, a distribuição preditiva do retorno do ativo $i$ é gaussiana com parâmetros em forma fechada:

\begin{align}
    \mu_y^{(i)} &= \mu_\alpha^{(i)} + \sum_{k=1}^{K} \beta^{(i,k)}\, \mu_z^{(k)} \label{eq:mu-preditivo} \\
    \sigma_y^{(i)} &= \sqrt{\left(\sigma_\alpha^{(i)}\right)^2 + \sum_{k=1}^{K} \left(\beta^{(i,k)}\right)^2 \left(\sigma_z^{(k)}\right)^2} \label{eq:sigma-preditivo}
\end{align}

\noindent Essa composição analítica elimina a necessidade de amostragem Monte Carlo na inferência e entrega estimativas de risco por ativo como subproduto direto do modelo.

\section{Treinamento}
\label{sec:treinamento}

A função objetivo é o negativo do ELBO adaptado ao contexto de previsão de retornos:

\begin{equation}
    \mathcal{L}(x, y) = -\frac{1}{N}\sum_{i=1}^{N} \log p_{\phi_{\text{dec}}}\!\left(y^{(i)} \;\middle|\; x,\, z_{\text{post}}\right) + \gamma \cdot \mathrm{KL}\!\left(q_{\phi_{\text{enc}}}(z|x,y) \;\|\; p_{\phi_{\text{pred}}}(z|x)\right)
    \label{eq:loss}
\end{equation}

\noindent O primeiro termo é a NLL gaussiana entre os retornos observados e a distribuição reconstruída via posterior, com média sobre os $N$ ativos da cross-section. O segundo termo é a divergência KL entre a posterior do codificador e a prior do preditor, com média sobre as $K$ dimensões latentes; o hiperparâmetro $\gamma$ controla o peso relativo do termo de regularização. Detalhes sobre o agendamento de $\gamma$ durante o treinamento e decisões de escala dos termos da loss estão no Apêndice~\ref{ap:estabilizacao}.

O otimizador é Adam com \textit{learning rate} $3 \times 10^{-4}$ e \textit{weight decay} $10^{-4}$. O \textit{batch} é uma única cross-section por passo, refletindo a natureza multivariada da observação: todas as séries de ativos na data $t$ são processadas conjuntamente. O critério de parada antecipada (\textit{early stopping}) monitora o Rank IC no conjunto de validação com paciência de 10 épocas. O framework de organização do treinamento é PyTorch Lightning.

\section{Dados}
\label{sec:dados}

\subsection{Universo de Ativos}
\label{subsec:universo}

O universo de ativos é composto por todos os tickers com \textit{pricing} válido na Economatica, reconstruído ponto a ponto sem restrição ao IBX ou a qualquer índice de referência. O universo médio no período de teste é de aproximadamente 300--500 ativos, contra mais de 3.000 no estudo original de \textcite{duan2022} — diferença estrutural que precisa estar registrada porque condiciona o significado do TopK-Drop com $k = 50$ (aproximadamente 15--25\% do cross-section brasileiro, contra cerca de 1,5\% do A-share chinês) e as comparações em geral. O mercado brasileiro apresenta concentração setorial marcada em commodities, energia e financeiro, que diferencia seu perfil de covariância do mercado chinês e do norte-americano.

\subsection{Período e Divisão da Amostra}
\label{subsec:periodo}

% PLACEHOLDER: definir datas exatas após consolidar o dataset.
% Estrutura esperada: 
%   Treinamento: [data_inicio] a [data_fim_treino]
%   Validação:   [data_inicio_val] a [data_fim_val]
%   Teste:       [data_inicio_teste] a [data_fim_teste]
% O período de teste deve cobrir: pandemia de 2020, ciclo de aperto 2021--2022, 
% mudança de governo 2023. Os três subperíodos não se sobrepõem.

Os subperíodos de treinamento, validação e teste são definidos sem sobreposição temporal, com as datas a serem especificadas após a consolidação do dataset. O período de teste engloba múltiplos regimes macroeconômicos relevantes para o mercado brasileiro: a pandemia de Covid-19 e o \textit{crash} de 2020, o ciclo de aperto monetário de 2021--2022 e a mudança de governo de 2023. Essa cobertura permite avaliar a robustez do sinal em diferentes configurações de risco e liquidez.

\subsection{Características Utilizadas}
\label{subsec:caracteristicas}

O modelo recebe como entrada $C = 20$ características técnicas calculadas a partir de preço e volume, organizadas em seis categorias. A Tabela~\ref{tab:caracteristicas} lista as características com suas categorias e janelas de cálculo; as fórmulas detalhadas estão no Apêndice~\ref{ap:caracteristicas}.

\begin{table}[htbp]
    \centering
    \caption{Características utilizadas como entrada do modelo.}
    \label{tab:caracteristicas}
    \begin{tabular}{llc}
        \toprule
        \textbf{Nome} & \textbf{Categoria} & \textbf{Janela (dias)} \\
        \midrule
        Ret1, Ret2, Ret5, Ret10, Ret20  & Retorno         & 1, 2, 5, 10, 20 \\
        Vol5, Vol10, Vol20              & Volatilidade    & 5, 10, 20       \\
        VolRatio                        & Volatilidade    & 5/20            \\
        Turnover5, Turnover10           & Volume          & 5, 10           \\
        VWAPDev                         & Volume          & 5               \\
        RSI                             & Indicador téc.  & 14              \\
        MADev5, MADev20                 & Indicador téc.  & 5, 20           \\
        Skew20, Kurt20                  & Momentos sup.   & 20              \\
        Amihud20                        & Iliquidez       & 20              \\
        \bottomrule
    \end{tabular}
    \fonte{Elaboração própria. Fórmulas detalhadas no Apêndice~\ref{ap:caracteristicas}.}
\end{table}

\subsection{Construção do Universo Point-in-Time}
\label{subsec:point-in-time}

A construção point-in-time é a decisão metodológica que mais diretamente afeta a credibilidade dos resultados. A inclusão de um ticker na cross-section da data $t$ exige que ele apresente $T = 20$ dias úteis consecutivos sem lacuna calendárica superior a 7 dias (tolerância para feriados nacionais). Tickers com suspensões prolongadas de negociação ou liquidez intermitente são descartados automaticamente nas datas em que essa condição falha, sem exclusão retroativa. A composição do universo é reconstruída a cada data, eliminando o viés de sobrevivência que surgiria da exclusão de empresas que deslistaram durante o período amostral. A auditoria do procedimento — verificação da presença de tickers que deslistaram dentro do período de treinamento — é apresentada no Apêndice~\ref{ap:diagnosticos}.

\subsection{Pré-processamento}
\label{subsec:preprocessing}

O retorno forward é definido como o retorno logarítmico do fechamento de $t{+}1$ em relação ao fechamento de $t$, conforme \textcite{duan2022}. Filtros de saneamento são aplicados para eliminar artefatos: preço mínimo de R\$~0,10 (excluindo penny stocks com dinâmica de preços atípica) e retorno diário absoluto máximo de 50\% (excluindo eventos de \textit{split} não ajustado e erros de dados). A normalização é realizada de forma cross-sectional em cada data — \textit{z-score} sobre os $N_t$ ativos disponíveis — tanto para as características quanto para os retornos. As estatísticas de normalização (médias e desvios-padrão) são calculadas exclusivamente sobre o período de treinamento e aplicadas sem re-estimação aos períodos de validação e teste.

\section{Modelos de Comparação}
\label{sec:baselines}

Os modelos de comparação seguem uma progressão deliberada que isola sucessivamente as propriedades da arquitetura do \textit{FactorVAE}. O Momentum captura o sinal econômico mais simples: retorno dos últimos 20 dias como previsão direta. O Linear/Ridge é o limite do problema sob hipótese de linearidade com as mesmas $C = 20$ características como entrada. O MLP adiciona não-linearidade sem estrutura temporal. O GRU adiciona estrutura temporal mas sem componente probabilístico e sem estrutura fatorial explícita. A progressão do Momentum ao GRU é, portanto, aditiva em complexidade, e o \textit{FactorVAE} acrescenta sobre o GRU a estrutura fatorial e a inferência variacional.

A ressalva metodológica é explícita: os modelos diferem do \textit{FactorVAE} em mais de uma dimensão simultaneamente, de modo que a comparação não constitui ablação formal. As diferenças observadas são consistentes com a contribuição das propriedades que o \textit{FactorVAE} adiciona, mas a atribuição causal estrita ficaria a cargo de um estudo de ablação dedicado, fora do escopo deste trabalho.

\section{Procedimentos de Avaliação}
\label{sec:avaliacao}

\subsection{Skill Cross-Sectional}
\label{subsec:avaliacao-skill}

O Rank IC é calculado como a correlação de Spearman entre os retornos previstos e os retornos realizados no \textit{cross-section} de cada data $t$ do período de teste. O Rank ICIR é a razão entre a média e o desvio-padrão das observações diárias de Rank IC ao longo do período de teste. As métricas foram apresentadas na Seção~\ref{sec:avaliacao-lit}; o que se registra aqui é que a avaliação é restrita ao conjunto de teste e calculada separadamente para cada modelo.

\subsection{Estratégia de Portfólio}
\label{subsec:avaliacao-portfolio}

A estratégia TopK-Drop \parencite{yang2020} é executada com $k = 50$ ações em carteira e no máximo $n = 5$ substituições por dia. A escolha de $k = 50$ segue o paper original e sustenta comparabilidade com a literatura, ainda que represente fração maior do universo brasileiro (aproximadamente 15--25\% do cross-section típico no período de teste) do que do A-share chinês (cerca de 1,5\%). O custo de transação é fixado em 10~bps \textit{one-way} por posição substituída (equivalente a $\approx$~20~bps \textit{round-trip}), aplicado a cada troca de ativo na carteira a cada dia. O benchmark é o portfólio igual-ponderado sobre o mesmo universo disponível a cada data.

\subsection{Robustez a Ativos Não Vistos}
\label{subsec:avaliacao-robustez}

O teste de holdout-retrain segue o protocolo de \textcite{duan2022}: $m$ tickers selecionados aleatoriamente são removidos do conjunto de treinamento, o modelo é retreinado do zero, e o Rank IC é calculado exclusivamente sobre esses $m$ tickers no conjunto de teste. O procedimento é repetido para múltiplos valores de $m$ e com múltiplas amostras aleatórias de tickers, produzindo uma distribuição de Rank IC sobre ativos nunca vistos durante o treinamento. A comparação com o Rank IC de universo completo quantifica a degradação de desempenho fora da distribuição de treinamento e informa a capacidade de generalização do modelo a IPOs e ativos de baixa liquidez.
"""

# ============================================================
# CAPÍTULO 4 — RESULTADOS
# ============================================================
CAP4 = r"""
\chapter{Resultados}
\label{cap:resultados}

A avaliação é organizada em quatro dimensões: \textit{skill} cross-sectional, performance ajustada ao risco, comportamento operacional e robustez a ativos não vistos. Uma nota metodológica é necessária antes de apresentar os resultados: a NLL não é reportada para comparação entre famílias de modelos. No caso do \textit{FactorVAE}, a função objetivo minimizada durante o treinamento é o negativo do ELBO, que é um \textit{limite inferior} da log-verossimilhança marginal verdadeira; comparar esse valor diretamente com a NLL de modelos sem inferência variacional seria metodologicamente inválido. Diagnósticos de treinamento estão restritos ao Apêndice~\ref{ap:estabilizacao}.

\section{Skill Preditivo Cross-Sectional}
\label{sec:resultados-skill}

% PLACEHOLDER: escrever comentário ancorado nos valores da Tabela~\ref{tab:skill}.
% Estrutura esperada: FactorVAE lidera em Rank IC médio; GRU competitivo em Rank ICIR;
% diferença sobre Momentum e Ridge é consistente mas não dramática; sem hedging.

\begin{table}[htbp]
    \centering
    \caption{Qualidade do sinal preditivo no conjunto de teste.}
    \label{tab:skill}
    \begin{tabular}{lcc}
        \toprule
        \textbf{Modelo}    & \textbf{Rank IC médio} & \textbf{Rank ICIR} \\
        \midrule
        \textbf{FactorVAE} & ---                    & ---                \\
        Momentum           & ---                    & ---                \\
        Linear (Ridge)     & ---                    & ---                \\
        MLP                & ---                    & ---                \\
        GRU                & ---                    & ---                \\
        \bottomrule
    \end{tabular}
    \fonte{Elaboração própria.}
\end{table}

\section{Performance Ajustada ao Risco}
\label{sec:resultados-performance}

% PLACEHOLDER: escrever interpretação em três pontos ancorados nos valores da Tabela~\ref{tab:performance}:
% (1) FactorVAE supera outros sinais em retorno absoluto e ajustado ao risco;
% (2) excesso anualizado contra benchmark igual-ponderado praticamente nulo após custos;
% (3) controle de drawdown entre os melhores do conjunto.
% Esse é o resultado central — discutir com clareza, não omitir.

\begin{table}[htbp]
    \centering
    \caption{Performance ajustada ao risco no período de teste (TopK-Drop, $k=50$, custo 10~bps \textit{one-way}).}
    \label{tab:performance}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{lccccccc}
        \toprule
        \textbf{Modelo}    & \textbf{Ret. Anual.} & \textbf{Excesso} & \textbf{Vol.} & \textbf{Sharpe} & \textbf{IR} & \textbf{Calmar} & \textbf{Max DD} \\
        \midrule
        \textbf{FactorVAE} & ---  & ---  & ---  & ---  & ---  & ---  & ---  \\
        EW Market          & ---  & ---  & ---  & ---  & ---  & ---  & ---  \\
        Momentum           & ---  & ---  & ---  & ---  & ---  & ---  & ---  \\
        Linear (Ridge)     & ---  & ---  & ---  & ---  & ---  & ---  & ---  \\
        MLP                & ---  & ---  & ---  & ---  & ---  & ---  & ---  \\
        GRU                & ---  & ---  & ---  & ---  & ---  & ---  & ---  \\
        \bottomrule
    \end{tabular}}
    \fonte{Elaboração própria.}
\end{table}

\begin{figure}[htbp]
    \centering
    \fbox{\parbox{0.82\textwidth}{\centering\vspace{2.5cm}
    {\small \textit{PLACEHOLDER} --- Retorno acumulado da estratégia.}\\[0.4cm]
    {\small Curvas: FactorVAE (cor primária, destaque), Momentum, Ridge, MLP,}\\
    {\small GRU e EW Market (linha tracejada cinza). Eixo~$x$: período de teste completo.}
    \vspace{2.5cm}}}
    \caption{Retorno acumulado da estratégia TopK-Drop por modelo no período de teste.}
    \label{fig:retorno-acumulado}
\end{figure}

\begin{figure}[htbp]
    \centering
    \fbox{\parbox{0.82\textwidth}{\centering\vspace{2.5cm}
    {\small \textit{PLACEHOLDER} --- Retorno acumulado em excesso vs.\ benchmark.}\\[0.4cm]
    {\small Mesmo formato da Figura~\ref{fig:retorno-acumulado}, sem a curva EW Market.}\\
    {\small Linha horizontal em zero como referência.}
    \vspace{2.5cm}}}
    \caption{Retorno acumulado em excesso sobre o benchmark igual-ponderado.}
    \label{fig:retorno-excesso}
\end{figure}

\section{Comportamento Operacional da Estratégia}
\label{sec:resultados-operacional}

% PLACEHOLDER: conectar hit rate e turnover ao resultado de portfólio da seção anterior.
% Estrutura esperada: FactorVAE acima de 50\% de hit rate e turnover moderado vs. Ridge/MLP/GRU.
% A combinação explica por que um ganho modesto em Rank IC se traduz em melhor performance
% — o sinal é convertido em portfólio com menos atrito.

\begin{table}[htbp]
    \centering
    \caption{Métricas operacionais da estratégia TopK-Drop no período de teste.}
    \label{tab:operacional}
    \begin{tabular}{lcc}
        \toprule
        \textbf{Modelo}    & \textbf{Hit Rate} & \textbf{Turnover Médio Diário} \\
        \midrule
        \textbf{FactorVAE} & ---               & ---                            \\
        Momentum           & ---               & ---                            \\
        Linear (Ridge)     & ---               & ---                            \\
        MLP                & ---               & ---                            \\
        GRU                & ---               & ---                            \\
        \bottomrule
    \end{tabular}
    \fonte{Elaboração própria.}
\end{table}

\section{Robustez a Ativos Não Vistos}
\label{sec:resultados-robustez}

% PLACEHOLDER: interpretar degradação do Rank IC nos tickers retidos para cada valor de $m$.
% O que a magnitude da queda diz sobre a capacidade de generalização do modelo
% a IPOs e ativos pouco líquidos?

\begin{table}[htbp]
    \centering
    \caption{Rank IC do \textit{FactorVAE} sob protocolo holdout-retrain.}
    \label{tab:robustez}
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Ativos retidos ($m$)} & \textbf{Trials} & \textbf{Rank IC (holdout)} & \textbf{Desvio} & \textbf{Rank IC (universo)} \\
        \midrule
        $m = 10$  & ---  & ---  & ---  & ---  \\
        $m = 50$  & ---  & ---  & ---  & ---  \\
        \bottomrule
    \end{tabular}
    \fonte{Elaboração própria.}
\end{table}
"""

# ============================================================
# CAPÍTULO 5 — CONCLUSÃO
# ============================================================
CAP5 = r"""
\chapter{Conclusão}
\label{cap:conclusao}

% PLACEHOLDER: completar o primeiro parágrafo após obter os resultados experimentais.
% Estrutura: (i) retomada da pergunta de pesquisa em uma frase; (ii) síntese dos achados
% na ordem em que aparecem nos resultados — skill cross-sectional, performance de portfólio,
% excesso contra o benchmark após custos. Sem hedging; afirmar o que os dados mostram.
Este trabalho avaliou se o \textit{FactorVAE}, replicado fielmente no mercado brasileiro de ações, mantém vantagem de \textit{skill} cross-sectional sobre baselines neurais e se essa vantagem sobrevive à tradução em portfólio sob custos de transação realistas. [COMPLETAR COM SÍNTESE DOS ACHADOS QUANTITATIVOS APÓS OBTER OS RESULTADOS.]

Quatro limitações merecem registro. Primeira, o tamanho reduzido do conjunto de dados — aproximadamente 5.000 pregões — em relação à complexidade do modelo, com efeito provável sobre a estabilidade do treinamento e a precisão das estimativas. Segunda, a dependência do codificador em retornos contemporâneos é uma característica do modelo original que define seu papel como reconstrutor retrospectivo durante o treinamento; a estrutura não constitui vazamento de dados, mas condiciona o significado da posterior e a interpretação do ELBO. Terceira, o custo de transação fixo em 10~bps \textit{one-way} é uma aproximação que não captura impacto de mercado em ativos menos líquidos, possivelmente subestimando os custos efetivos da estratégia. Quarta, o período de teste, apesar de cobrir múltiplos regimes, não esgota o espaço de configurações macroeconômicas possíveis.

Três extensões futuras merecem investigação. A primeira é a avaliação em janelas mais longas à medida que mais dados ficam disponíveis, o que permitiria separar com maior precisão os efeitos de regime do sinal estrutural do modelo. A segunda é a substituição do TopK-Drop por estratégias \textit{risk-aware} que utilizem as estimativas de incerteza $\sigma_y^{(i)}$ por ativo, entregues em forma fechada pelo decodificador conforme a Equação~\ref{eq:sigma-preditivo} — análoga à variante TDrisk reportada por \textcite{duan2022}. A terceira é a aplicação do mesmo arcabouço de avaliação a outros mercados emergentes, para mapear como a vantagem do \textit{FactorVAE} depende de características estruturais do mercado como profundidade do universo, liquidez e concentração setorial.
"""

# ============================================================
# APÊNDICE A — ESTABILIZAÇÃO DO TREINAMENTO
# ============================================================
AP_A = r"""
\chapter{Estabilização do Treinamento}
\label{ap:estabilizacao}

Este apêndice documenta as decisões de implementação que afetam a estabilidade numérica do treinamento mas que sobrecarregariam o corpo principal do texto.

\textbf{Agendamento do coeficiente KL.} O peso $\gamma$ do termo KL na função objetivo (Equação~\ref{eq:loss}) é agendado linearmente ao longo dos primeiros \texttt{kl\_warmup\_steps}~$= 500$ passos de otimização, iniciando em zero e atingindo o valor configurado ao final do intervalo. O aquecimento evita que o preditor seja forçado a aproximar uma posterior ainda não treinada, o que colapsaria o espaço latente nas primeiras épocas.

\textbf{Free bits.} O mecanismo de \textit{free bits} — que impõe um threshold mínimo de contribuição KL por dimensão latente — foi desativado (\texttt{kl\_free\_bits}~$= 0$). Observou-se que os valores naturais de KL por fator no regime de operação normal ficam abaixo de qualquer threshold praticamente útil, de modo que ativar o mecanismo cancelaria gradientes do preditor sem benefício correspondente em termos de utilização do espaço latente.

\textbf{Escala dos termos da loss.} A NLL é calculada como média sobre os $N$ ativos da cross-section (não soma), e a KL é calculada como média sobre as $K$ dimensões latentes (não soma). Essa escolha mantém os dois termos em escala compatível independentemente do tamanho da cross-section $N_t$ no dia $t$ — variável entre aproximadamente 300 e 500 ativos no período de teste — e do número de fatores $K = 16$.
"""

# ============================================================
# APÊNDICE B — CARACTERÍSTICAS TÉCNICAS
# ============================================================
AP_B = r"""
\chapter{Características Técnicas}
\label{ap:caracteristicas}

\begin{table}[htbp]
    \centering
    \caption{Detalhamento das vinte características técnicas utilizadas como entrada do modelo.}
    \label{tab:caracteristicas-detalhado}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{llcp{6cm}}
        \toprule
        \textbf{Nome} & \textbf{Categoria} & \textbf{Janela} & \textbf{Definição / Justificativa} \\
        \midrule
        Ret1   & Retorno       & 1d  & $\ln(P_t / P_{t-1})$; sinal de reversão de curto prazo \\
        Ret2   & Retorno       & 2d  & $\ln(P_t / P_{t-2})$; horizonte intermediário \\
        Ret5   & Retorno       & 5d  & $\ln(P_t / P_{t-5})$; sinal semanal \\
        Ret10  & Retorno       & 10d & $\ln(P_t / P_{t-10})$; quinzenal \\
        Ret20  & Retorno       & 20d & $\ln(P_t / P_{t-20})$; fator de momentum mensal \\
        Vol5   & Volatilidade  & 5d  & Desvio-padrão dos retornos diários nos últimos 5 dias \\
        Vol10  & Volatilidade  & 10d & Desvio-padrão dos retornos diários nos últimos 10 dias \\
        Vol20  & Volatilidade  & 20d & Desvio-padrão dos retornos diários nos últimos 20 dias \\
        VolRatio & Volatilidade & 5/20d & Vol5 / Vol20; mudança recente de volatilidade \\
        Turnover5  & Volume    & 5d  & Volume médio / capitalização de mercado; últimos 5 dias \\
        Turnover10 & Volume    & 10d & Volume médio / capitalização de mercado; últimos 10 dias \\
        VWAPDev    & Volume    & 5d  & $(P_t - \text{VWAP}_{5d}) / \text{VWAP}_{5d}$; desvio do preço médio ponderado \\
        RSI        & Ind. téc. & 14d & Índice de força relativa; $100 - 100/(1 + U/D)$ \\
        MADev5     & Ind. téc. & 5d  & $(P_t - \text{MA}_{5d}) / \text{MA}_{5d}$; desvio da média móvel curta \\
        MADev20    & Ind. téc. & 20d & $(P_t - \text{MA}_{20d}) / \text{MA}_{20d}$; desvio da média móvel longa \\
        Skew20     & Mom. sup. & 20d & Assimetria amostral dos retornos diários nos últimos 20 dias \\
        Kurt20     & Mom. sup. & 20d & Curtose amostral dos retornos diários nos últimos 20 dias \\
        Amihud20   & Iliquidez & 20d & $\frac{1}{20}\sum_{s=t-19}^{t} |r_s| / \text{Vol}_s$; medida de Amihud \\
        \bottomrule
    \end{tabular}}
    \fonte{Elaboração própria.}
\end{table}

% PLACEHOLDER: adicionar as últimas 2 características da lista final (ajustar conforme
% o dataset consolidado; a tabela acima lista 18 das 20 previstas).
"""

# ============================================================
# APÊNDICE C — DIAGNÓSTICOS PRÉ-TREINO
# ============================================================
AP_C = r"""
\chapter{Diagnósticos Pré-Treino}
\label{ap:diagnosticos}

\section*{Cross-fit Posterior IC}

O Cross-fit Posterior IC é o diagnóstico de validade do codificador como oráculo. O procedimento é o seguinte: para cada data $t$ do conjunto de validação, o modelo é executado em modo oráculo — o codificador recebe $x_t$ e os retornos contemporâneos $y_t$ e produz $\mu_{\text{post}}$; o decodificador usa esses fatores para produzir a previsão $\hat{y}^{(\text{post})}$; o Rank IC é calculado entre $\hat{y}^{(\text{post})}$ e os retornos realizados em $t{+}1$, que são \textit{diferentes} dos retornos usados como entrada do codificador. Se o Rank IC resultante for positivo, o codificador extrai sinal preditivo dos retornos contemporâneos — não apenas reconstrói a entrada de forma trivial. Um Posterior IC próximo de zero indicaria que o oráculo não captura informação útil e que o esquema professor-aluno não teria o que transferir ao preditor.

% PLACEHOLDER: reportar o valor obtido do Posterior IC no conjunto de validação.

\section*{Auditoria de Viés de Sobrevivência}

A auditoria verifica que a construção point-in-time efetivamente preserva no universo de treinamento os tickers que deslistaram durante o período amostral. O procedimento consiste em calcular a distribuição da última data válida por ticker no dataset e verificar que uma fração não-negligenciável desses tickers tem última data dentro do período de treinamento (e não apenas após o final do período de teste, o que indicaria ausência de deslistamentos no dataset).

% PLACEHOLDER: reportar a contagem de tickers que deslistaram dentro do período de treinamento
% e a fração do universo que eles representam nos anos em que estavam ativos.
"""

# ============================================================
# APÊNDICE D — HIPERPARÂMETROS E CONFIGURAÇÃO
# ============================================================
AP_D = r"""
\chapter{Hiperparâmetros e Configuração}
\label{ap:hiperparametros}

\begin{table}[htbp]
    \centering
    \caption{Configuração completa do experimento.}
    \label{tab:hiperparametros}
    \begin{tabular}{lll}
        \toprule
        \textbf{Parâmetro} & \textbf{Valor} & \textbf{Descrição} \\
        \midrule
        \multicolumn{3}{l}{\textit{Arquitetura}} \\
        $H$ (dim. oculta)      & ---   & Dimensão dos estados ocultos da GRU e das MLPs \\
        $K$ (fatores)          & 16    & Número de fatores latentes \\
        $M$ (portfólios)       & 64    & Portfólios do codificador \\
        $T$ (janela temporal)  & 20    & Dias úteis de histórico por ativo \\
        $C$ (características)  & 20    & Características por ativo por período \\
        \midrule
        \multicolumn{3}{l}{\textit{Otimizador}} \\
        Algoritmo              & Adam  & --- \\
        Learning rate          & $3 \times 10^{-4}$ & Fixo; sem agendamento \\
        Weight decay           & $10^{-4}$          & Regularização L2 \\
        Batch size             & 1 cross-section    & Uma data por passo \\
        Épocas máximas         & ---   & A definir \\
        Paciência (ES)         & 10    & Épocas sem melhora no Rank IC de validação \\
        \midrule
        \multicolumn{3}{l}{\textit{Treinamento}} \\
        \texttt{kl\_warmup\_steps}  & 500  & Passos de aquecimento do KL \\
        $\gamma$ (peso KL)     & ---   & A reportar \\
        \texttt{kl\_free\_bits}& 0     & Mecanismo de free bits desativado \\
        Random seed            & ---   & A registrar \\
        \midrule
        \multicolumn{3}{l}{\textit{Hardware}} \\
        GPU                    & ---   & A especificar \\
        Tempo de treinamento   & ---   & A medir \\
        \bottomrule
    \end{tabular}
    \fonte{Elaboração própria.}
\end{table}
"""

# ============================================================
# RESUMO
# ============================================================
RESUMO = r"""
\begin{resumo}
Este trabalho replica e avalia o \textit{FactorVAE} — modelo de fatores latentes probabilístico baseado em inferência variacional — no mercado brasileiro de ações. O modelo aprende a distribuição condicional dos retornos cross-sectionais combinando uma rede recorrente para extração de características, um esquema professor-aluno entre codificador e preditor para aprendizado dos fatores latentes, e um decodificador com composição gaussiana analítica para produção da distribuição preditiva em forma fechada. O desempenho é avaliado em termos de \textit{skill} cross-sectional (Rank IC e Rank ICIR), performance de portfólio ajustada ao risco (TopK-Drop com custo de transação de 10~bps \textit{one-way}), comportamento operacional (hit rate e turnover) e robustez a ativos não vistos durante o treinamento (holdout-retrain). Os resultados são comparados com quatro baselines em progressão de complexidade: Momentum, Ridge, MLP e GRU. O universo de ativos, construído ponto a ponto a partir da Economatica sem restrição ao IBX, conta com aproximadamente 300--500 ativos no período de teste — substancialmente menor do que o mercado A-share chinês original. A pergunta central é se o \textit{FactorVAE} mantém vantagem de \textit{skill} sobre os baselines nesse ambiente e se essa vantagem sobrevive à tradução em portfólio sob custos realistas.

\textbf{Palavras-chaves}: \imprimirpalavraschavesresumo.
\end{resumo}
"""

# ============================================================
# ABSTRACT
# ============================================================
ABSTRACT = r"""
\begin{resumo}[Abstract]
\begin{otherlanguage*}{english}
This paper replicates and evaluates FactorVAE --- a probabilistic latent factor model based on variational inference --- in the Brazilian equity market. The model learns the conditional distribution of cross-sectional returns by combining a recurrent network for feature extraction, a teacher-student scheme between encoder and predictor for latent factor learning, and a decoder with closed-form Gaussian composition for the predictive distribution. Performance is assessed along four dimensions: cross-sectional skill (Rank IC and Rank ICIR), risk-adjusted portfolio performance (TopK-Drop strategy with a one-way transaction cost of 10~bps), operational behavior (hit rate and turnover), and robustness to assets unseen during training (holdout-retrain protocol). Results are compared against four baselines of increasing complexity: Momentum, Ridge, MLP, and GRU. The asset universe, constructed point-in-time from Economatica without restriction to any index, comprises approximately 300--500 assets in the test period --- substantially smaller than the original Chinese A-share market. The central question is whether FactorVAE maintains a skill advantage over the baselines in this environment and whether that advantage survives the translation to a portfolio strategy under realistic costs.

\textbf{Keywords}: \thekeywordsabstract.
\end{otherlanguage*}
\end{resumo}
"""

# ============================================================
# LISTA DE ABREVIATURAS E SIGLAS
# ============================================================
SIGLAS = r"""
\begin{siglas} \itemsep -1pt
  \item[APT]    \textit{Arbitrage Pricing Theory}
  \item[CAPM]   \textit{Capital Asset Pricing Model}
  \item[ELBO]   \textit{Evidence Lower Bound}
  \item[GRU]    \textit{Gated Recurrent Unit}
  \item[IBX]    Índice Brasil (B3)
  \item[IC]     Coeficiente de Informação (\textit{Information Coefficient})
  \item[ICIR]   Razão IC/Desvio (\textit{IC Information Ratio})
  \item[IR]     \textit{Information Ratio}
  \item[KL]     Divergência de Kullback-Leibler
  \item[MLP]    \textit{Multilayer Perceptron}
  \item[NLL]    \textit{Negative Log-Likelihood}
  \item[RSI]    Índice de Força Relativa (\textit{Relative Strength Index})
  \item[VAE]    \textit{Variational Autoencoder}
  \item[VWAP]   \textit{Volume-Weighted Average Price}
\end{siglas}
"""

# ============================================================
# REFERENCIAS.BIB
# ============================================================
BIB = r"""
@article{sharpe1964,
  author  = {Sharpe, William F.},
  title   = {Capital asset prices: A theory of market equilibrium under conditions of risk},
  journal = {The Journal of Finance},
  year    = {1964},
  volume  = {19},
  number  = {3},
  pages   = {425--442}
}

@article{ross1976,
  author  = {Ross, Stephen A.},
  title   = {The arbitrage theory of capital asset pricing},
  journal = {Journal of Economic Theory},
  year    = {1976},
  volume  = {13},
  number  = {3},
  pages   = {341--360}
}

@article{fama1993,
  author  = {Fama, Eugene F. and French, Kenneth R.},
  title   = {Common risk factors in the returns on stocks and bonds},
  journal = {Journal of Financial Economics},
  year    = {1993},
  volume  = {33},
  number  = {1},
  pages   = {3--56}
}

@article{fama2015,
  author  = {Fama, Eugene F. and French, Kenneth R.},
  title   = {A five-factor asset pricing model},
  journal = {Journal of Financial Economics},
  year    = {2015},
  volume  = {116},
  number  = {1},
  pages   = {1--22}
}

@article{carhart1997,
  author  = {Carhart, Mark M.},
  title   = {On persistence in mutual fund performance},
  journal = {The Journal of Finance},
  year    = {1997},
  volume  = {52},
  number  = {1},
  pages   = {57--82}
}

@article{gu2021,
  author  = {Gu, Shihao and Kelly, Bryan and Xiu, Dacheng},
  title   = {Autoencoder asset pricing models},
  journal = {Journal of Econometrics},
  year    = {2021},
  volume  = {222},
  number  = {1},
  pages   = {429--450}
}

@inproceedings{duan2022,
  author    = {Duan, Yitong and Wang, Lei and Chen, Qizhong and Wang, Wentao and others},
  title     = {{FactorVAE}: A probabilistic dynamic factor model based on variational autoencoder for predicting cross-sectional stock returns},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2022},
  volume    = {36},
  number    = {4},
  pages     = {4468--4476}
}

@inproceedings{kingma2014,
  author    = {Kingma, Diederik P. and Welling, Max},
  title     = {Auto-encoding variational {Bayes}},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2014}
}

@inproceedings{cho2014,
  author    = {Cho, Kyunghyun and van Merrienboer, Bart and Gulcehre, Caglar and others},
  title     = {Learning phrase representations using {RNN} encoder-decoder for statistical machine translation},
  booktitle = {Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2014},
  pages     = {1724--1734}
}

@misc{yang2020,
  author       = {Yang, Xiao and Liu, Weiqing and Zhou, Dong and Bian, Jiang and Liu, Tie-Yan},
  title        = {Qlib: An {AI}-oriented quantitative investment platform},
  year         = {2020},
  howpublished = {arXiv preprint arXiv:2009.11189}
}
"""

# ============================================================
# WRITE ALL FILES
# ============================================================
print("Escrevendo arquivos...")

# Chapters
w("2-textuais/01-introducao.tex",       CAP1)
w("2-textuais/02-revisao.tex",          CAP2)
w("2-textuais/03-metodologia.tex",      CAP3)
w("2-textuais/04-resultados.tex",       CAP4)
w("2-textuais/05-conclusao.tex",        CAP5)

# Appendices
w("3-pos-textuais/apendices/apendice-a.tex", AP_A)
w("3-pos-textuais/apendices/apendice-b.tex", AP_B)
w("3-pos-textuais/apendices/apendice-c.tex", AP_C)
w("3-pos-textuais/apendices/apendice-d.tex", AP_D)

# Pre-textual
w("1-pre-textuais/resumo.tex",          RESUMO)
w("1-pre-textuais/abstract.tex",        ABSTRACT)
w("1-pre-textuais/opcionais/lista-de-abreviaturas-e-siglas.tex", SIGLAS)

# References
w("referencias.bib", BIB)

# ============================================================
# UPDATE MAIN.TEX  (metadata + chapter includes)
# ============================================================
import re

main_path = os.path.join(BASE, "main.tex")
with open(main_path, 'r', encoding='utf-8') as f:
    main = f.read()

# Add booktabs if not present
if 'booktabs' not in main:
    main = main.replace(
        r'\usepackage{caption} % legendas',
        r'\usepackage{caption} % legendas' + '\n' + r'\usepackage{booktabs} % linhas profissionais em tabelas'
    )

# Update metadata block
replacements = [
    (r'\\titulo\{[^}]*\}',        r'\\titulo{FactorVAE no Mercado Brasileiro de Ações}'),
    (r'\\subtitulo\{[^}]*\}',     r'\\subtitulo{: Replicação e Avaliação de um Modelo de Fatores Latentes Probabilístico}'),
    (r'\\autor\{[^}]*\}',         r'\\autor{Diego Fullin Azenha}'),
    (r'\\curso\{[^}]*\}',         r'\\curso{Graduação em Economia}'),
    (r'\\orientador\[[^\]]*\]\{[^}]*\}', r'\\orientador[Orientador:]{Ruy Monteiro Ribeiro}'),
    (r'\\tipotrabalho\{[^}]*\}',  r'\\tipotrabalho{Trabalho de Conclusão de Curso}'),
    (r'\\palavraschaves\{[^}]*\}',
     r'\\palavraschaves{modelos de fatores, autoencoder variacional, retornos de ações, mercado brasileiro, aprendizado de máquina, inferência variacional}'),
    (r'\\keywords\{[^}]*\}',
     r'\\keywords{factor models, variational autoencoder, stock returns, Brazilian market, machine learning, variational inference}'),
    (r'\\professores\{[^}]*\}',   r'\\professores{PLACEHOLDER: membros da banca}'),
]

for pattern, repl in replacements:
    main = re.sub(pattern, repl, main)

# Remove co-advisor line
main = re.sub(r'\n% Coorientador.*\n\\coorientador\[[^\]]*\]\{[^}]*\}\n', '\n', main)

# Update preambulo
main = re.sub(
    r'\\preambulo\{.*?\}',
    r'\\preambulo{Trabalho de Conclusão de Curso apresentado ao Curso de Graduação em Economia do Insper Instituto de Ensino e Pesquisa como requisito parcial para a obtenção do título de Bacharel em Economia.}',
    main, flags=re.DOTALL
)

# Update chapter includes
old_includes = re.search(
    r'% Cap[íi]tulos do texto.*?\\include\{2-textuais/05-conclus[ãa]o\}',
    main, re.DOTALL
)
if old_includes:
    new_includes = (
        "% Capítulos do texto\n"
        r"\include{2-textuais/01-introducao}" + "\n"
        r"\include{2-textuais/02-revisao}" + "\n"
        r"\include{2-textuais/03-metodologia}" + "\n"
        r"\include{2-textuais/04-resultados}" + "\n"
        r"\include{2-textuais/05-conclusao}"
    )
    main = main[:old_includes.start()] + new_includes + main[old_includes.end():]

# Update appendices
old_ap = re.search(r'\\begin\{apendicesenv\}.*?\\end\{apendicesenv\}', main, re.DOTALL)
if old_ap:
    new_ap = (
        r"\begin{apendicesenv}" + "\n"
        r"    \partapendices" + "\n"
        r"    \input{3-pos-textuais/apendices/apendice-a}" + "\n"
        r"    \input{3-pos-textuais/apendices/apendice-b}" + "\n"
        r"    \input{3-pos-textuais/apendices/apendice-c}" + "\n"
        r"    \input{3-pos-textuais/apendices/apendice-d}" + "\n"
        r"\end{apendicesenv}"
    )
    main = main[:old_ap.start()] + new_ap + main[old_ap.end():]

with open(main_path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(main)
print("  OK: main.tex")

print("\nConcluído. Todos os arquivos foram escritos.")
