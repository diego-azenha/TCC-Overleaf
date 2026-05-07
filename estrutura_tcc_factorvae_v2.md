# Estruturação do TCC — FactorVAE aplicado ao mercado brasileiro

**Aluno:** Diego Fullin Azenha
**Orientador:** Ruy Monteiro Ribeiro
**Curso:** Graduação em Economia — Insper

---

## Scaffolding completo

```
1  INTRODUÇÃO

2  REVISÃO DA LITERATURA
   2.1  Modelos de Fatores: dos Observáveis aos Latentes
   2.2  Variational Autoencoders
   2.3  O Modelo FactorVAE
   2.4  Avaliação de Modelos Preditivos em Finanças

3  METODOLOGIA
   3.1  Formulação do Problema
   3.2  Estrutura do Modelo
        3.2.1  Extrator de Características
        3.2.2  Codificador de Fatores
        3.2.3  Preditor de Fatores
        3.2.4  Decodificador de Fatores
   3.3  Treinamento
   3.4  Dados
        3.4.1  Universo de Ativos
        3.4.2  Período e Divisão da Amostra
        3.4.3  Características Utilizadas
        3.4.4  Construção do Universo Point-in-Time
        3.4.5  Pré-processamento
   3.5  Modelos de Comparação
   3.6  Procedimentos de Avaliação
        3.6.1  Skill Cross-Sectional
        3.6.2  Estratégia de Portfólio
        3.6.3  Robustez a Ativos Não Vistos

4  RESULTADOS
   4.1  Skill Preditivo Cross-Sectional
   4.2  Performance Ajustada ao Risco
   4.3  Comportamento Operacional da Estratégia
   4.4  Robustez a Ativos Não Vistos

5  CONCLUSÃO

APÊNDICES
   A  Estabilização do Treinamento
   B  Características Técnicas
   C  Diagnósticos Pré-Treino
   D  Hiperparâmetros e Configuração
```

---

## 1 Introdução

Funil em três movimentos, no estilo da entrega intermediária. Sem fórmulas — a equação do CAPM e demais ferramentas formais ficam reservadas para a revisão e a metodologia.

**Movimento 1 — Contextualização.** A função econômica dos modelos de fatores: decompor risco, estimar covariância em cross-section, formar expectativas condicionais de retorno como combinação de exposições sistemáticas. A sequência canônica situa o leitor sem aprofundar — Sharpe (1964), Ross (1976), Fama e French (1993, 2015).

**Movimento 2 — Virada metodológica.** Modelos clássicos definem fatores ex-ante e estimam exposições por regressão linear. Modelos data-driven aprendem fatores diretamente dos dados, ao custo de operar sob baixa razão sinal-ruído. A virada para abordagens neurais probabilísticas surge como tentativa de tratar esse ruído de forma explícita, modelando a distribuição condicional dos retornos em vez de estimativas pontuais. Citações: Gu, Kelly e Xiu (2021) para a referência neural não-probabilística; Duan et al. (2022) para o FactorVAE.

**Movimento 3 — Lacuna e pergunta de pesquisa.** O FactorVAE foi avaliado no mercado A-share chinês, com mais de três mil ativos, regime de liquidez próprio e período de teste de dois anos. O comportamento do modelo em mercados emergentes menores, com algumas centenas de ativos, custos de transação não-triviais e período de teste mais longo, não foi documentado. A pergunta é direta: o FactorVAE replicado fielmente no mercado brasileiro mantém vantagem de skill cross-sectional sobre baselines neurais, e essa vantagem sobrevive à tradução em portfólio sob custos realistas?

**Fechamento.** Parágrafo curto encadeando os capítulos.

> **Sem figuras, tabelas ou fórmulas nesta seção.**

---

## 2 Revisão da Literatura

A revisão é deliberadamente enxuta. O capítulo entrega apenas o que o leitor precisa para acompanhar a metodologia: o lugar do FactorVAE na evolução da modelagem fatorial, o vocabulário de inferência variacional necessário para entender sua função objetivo, a apresentação do modelo em si, e os procedimentos de avaliação que serão aplicados aos resultados. Histórico exaustivo da teoria de fatores fica fora.

### 2.1 Modelos de Fatores: dos Observáveis aos Latentes

Subseção única, organizada em quatro parágrafos. O primeiro parágrafo apresenta a estrutura geral dos modelos fatoriais — retornos como combinação de exposições a fatores comuns mais um componente idiossincrático — citando CAPM (Sharpe 1964) e APT (Ross 1976) como pontos de referência sem reproduzir derivações.

O segundo parágrafo cobre fatores observáveis construídos a partir de características: Fama e French (1993, 2015), Carhart (1997). O ponto a fixar é a lógica de portfólios formados ex-ante a partir de hipóteses sobre quais características geram prêmios de risco, e a estimação por regressão linear de painel.

O terceiro parágrafo apresenta os limites estruturais dessa abordagem em três pontos curtos: dependência da especificação ex-ante, restrição da hipótese linear, e o factor zoo. Esses limites motivam a busca por modelos que aprendam fatores diretamente dos dados.

O quarto parágrafo introduz o Conditional Autoencoder de Gu, Kelly e Xiu (2021) como exemplo de modelo neural com fatores latentes não-probabilísticos, e termina situando o FactorVAE como extensão probabilística dessa linha — preparação direta para a apresentação detalhada na subseção 2.3.

> **Fórmula 2.1 — Modelo fatorial geral:** $r_{i,t} = \alpha_i + \sum_k \beta_{i,k} f_{k,t} + \varepsilon_{i,t}$, apresentada como referência conceitual antes de discutir variantes.

### 2.2 Variational Autoencoders

Subseção curta, focada estritamente no que sustenta a função objetivo do FactorVAE. Três parágrafos.

Primeiro parágrafo apresenta o problema central: a verossimilhança marginal $p(x) = \int p(x|z) p(z) \, dz$ não tem forma fechada quando o decoder é não-linear, o que inviabiliza estimação direta por máxima verossimilhança.

Segundo parágrafo apresenta a inferência variacional como solução: introduz-se uma distribuição aproximada $q(z|x)$ e maximiza-se um limite inferior da log-verossimilhança — o ELBO — composto por um termo de reconstrução e um termo KL. A reparametrização (Kingma e Welling 2014) é mencionada como mecanismo que permite gradiente através do sampling.

Terceiro parágrafo distingue VAEs com prior fixa $\mathcal{N}(0, I)$ de VAEs com prior aprendida. No segundo caso, o termo KL aproxima duas distribuições aprendidas, e o comportamento numérico do treinamento muda. Esse ponto antecipa a estrutura específica do FactorVAE, no qual o codificador produz uma posterior e o preditor produz uma prior, ambas aprendidas.

> **Fórmula 2.2 — ELBO:** $\log p(x) \geq \mathbb{E}_q[\log p(x|z)] - \text{KL}(q(z|x) \,\|\, p(z))$, com explicação dos dois termos no parágrafo subsequente.

### 2.3 O Modelo FactorVAE

Apresentação textual da estrutura proposta por Duan et al. (2022). A descrição precisa cobrir três pontos, todos formulados de modo a evitar paráfrase próxima do paper original.

O primeiro ponto é a decomposição econômica que o modelo preserva: cada retorno é uma soma de componente idiossincrático e componente sistemático, este último escrito como combinação linear de fatores latentes. A inovação do FactorVAE não está na decomposição, e sim no fato de que todos os componentes — alpha, exposições e fatores — são saídas de redes neurais que processam a informação histórica.

O segundo ponto é o esquema professor-aluno, que distingue o FactorVAE de um VAE convencional. Durante o treinamento, o codificador recebe os retornos contemporâneos como sinal e produz uma distribuição posterior sobre os fatores; o preditor recebe apenas informação histórica e produz uma distribuição prior. A função objetivo aproxima a prior da posterior. Como consequência direta, o codificador é descartado na inferência: apenas o preditor e o decodificador participam da previsão fora da amostra. Esse ponto é o que sustenta toda a arquitetura e precisa estar absolutamente claro para o leitor.

O terceiro ponto é a composição analítica gaussiana no decodificador, que entrega distribuição preditiva fechada para os retornos sem amostragem Monte Carlo — propriedade que será aproveitada na construção das estimativas de risco do modelo.

> **🖼️ Figura 1 — Diagrama geral da arquitetura do FactorVAE.** Quatro blocos (Extrator de Características, Codificador, Preditor, Decodificador) e o fluxo de dados entre eles, com indicação visual explícita de quais blocos participam apenas do treinamento (codificador, com entrada de retornos contemporâneos) e quais participam também da inferência (extrator, preditor, decodificador). Adaptada de Duan et al. (2022). Posição: imediatamente após o parágrafo que descreve a decomposição econômica, antes da explicação do esquema professor-aluno.

### 2.4 Avaliação de Modelos Preditivos em Finanças

Subseção que apresenta as métricas e procedimentos a serem aplicados no capítulo de resultados, para que o leitor não encontre ferramentas pela primeira vez no meio da análise. Três parágrafos.

Primeiro parágrafo apresenta o Rank IC como correlação de Spearman entre retornos previstos e realizados em cross-section, calculada por data, e o Rank ICIR como razão entre média e desvio do Rank IC ao longo do tempo, captando consistência do sinal.

Segundo parágrafo apresenta a estratégia TopK-Drop (Yang et al. 2020) como tradução padrão de skill em portfólio: selecionar $k$ ativos com maior retorno previsto, permitindo no máximo $n$ substituições por dia. As métricas usuais de portfólio (Sharpe, Information Ratio, Calmar, drawdown, hit rate, turnover) são listadas com uma frase explicando o que cada uma captura.

Terceiro parágrafo trata da disciplina amostral. Viés de sobrevivência e a importância da construção point-in-time do universo, splits temporais sem sobreposição, normalização restrita ao período de treinamento. Esses são pontos que serão cobrados nas decisões metodológicas do capítulo seguinte.

> **Sem figuras, tabelas ou fórmulas adicionais nesta subseção.**

---

## 3 Metodologia

Parágrafo de abertura curto: a transição da estrutura conceitual apresentada na revisão para a especificação operacional do modelo, com ênfase no que precisa ser definido para que o trabalho seja replicável.

### 3.1 Formulação do Problema

Apresentação matemática da tarefa de modelagem. O modelo busca a distribuição condicional dos retornos cross-sectionais dado o conjunto de informações até $t$. A decomposição fatorial dinâmica tem todos os componentes como funções aprendidas das características históricas.

Os parâmetros de dimensão são explicitados com seus valores: $N$ (variável no tempo, cerca de 300–500 ativos no período de teste), $T = 20$ dias, $C = 20$ características, $K = 16$ fatores latentes, $M = 64$ portfólios construídos no codificador. Cada quantidade com uma frase de justificativa curta, evitando reprodução do paper original.

> **Fórmula 3.1 — Decomposição fatorial dinâmica:** $r_{t+1} = \alpha(x_t) + \beta(x_t) z_{t+1} + \varepsilon_{t+1}$, com $x_t \in \mathbb{R}^{N \times T \times C}$.

### 3.2 Estrutura do Modelo

Parágrafo de abertura: a arquitetura é organizada em quatro blocos. Cada bloco recebe uma subseção curta, focada em explicar a função do bloco com o mínimo de fórmulas necessário.

> **🖼️ Figura 2 — Diagrama detalhado da arquitetura.** Versão mais granular da Figura 1, mostrando as camadas internas de cada bloco e as dimensões dos tensores trocados. Elaboração própria.

**3.2.1 Extrator de Características.** GRU (Cho et al. 2014) sobre janela de $T = 20$ dias, com projeção linear inicial seguida de LeakyReLU. O último estado oculto da GRU é a representação por ativo $e^{(i)} \in \mathbb{R}^H$, com pesos compartilhados entre ativos. Sem fórmulas — a operação é padrão e a descrição textual é suficiente.

**3.2.2 Codificador de Fatores.** Três parágrafos, com cuidado especial para precisão e originalidade na descrição.

Primeiro parágrafo descreve a PortfolioLayer. As representações por ativo $e^{(i)}$ parametrizam pesos de portfólio através de uma camada linear seguida de softmax sobre ativos. Os pesos vêm das representações, não dos retornos. Os retornos contemporâneos entram apenas como o sinal a ser ponderado por esses pesos, produzindo $M$ retornos de portfólio.

Segundo parágrafo descreve a MappingLayer: os retornos de portfólio são projetados nos parâmetros da posterior $(\mu_{\text{post}}, \sigma_{\text{post}})$ via duas camadas lineares, com Softplus garantindo positividade do desvio.

Terceiro parágrafo fixa a interpretação. O codificador funciona como oráculo via os retornos realizados, mas a alocação que define como esses retornos são agregados é informada pelas representações. Essa distinção é o que dá conteúdo ao esquema professor-aluno: as representações por ativo determinam onde o sinal contemporâneo é coletado, e o sinal coletado parametriza a posterior. A validade do codificador como oráculo é confirmada pelo diagnóstico de cross-fit Posterior IC reportado no Apêndice C.

> **Fórmula 3.2 — Pesos de portfólio:** $a^{(i,j)} = \frac{\exp(W_p e^{(i)} + b_p)^{(j)}}{\sum_{i'} \exp(W_p e^{(i')} + b_p)^{(j)}}$, softmax sobre ativos para cada um dos $M$ portfólios.

> **Fórmula 3.3 — Agregação dos retornos pelos pesos:** $y_p^{(j)} = \sum_i y^{(i)} a^{(i,j)}$, definindo o retorno do $j$-ésimo portfólio que será projetado na posterior.

**3.2.3 Preditor de Fatores.** $K$ cabeças de atenção independentes, cada uma com query aprendida $q \in \mathbb{R}^H$, similaridade cosseno entre query e chaves, e gate ReLU. Cada cabeça produz uma representação global do mercado a partir das representações por ativo. Uma DistributionNetwork compartilhada entre cabeças produz os parâmetros da prior $(\mu_{\text{prior}}, \sigma_{\text{prior}})$. A descrição textual basta para o leitor entender o mecanismo; uma única fórmula formaliza o ponto não-trivial — a ausência de softmax e a presença do gate ReLU.

> **Fórmula 3.4 — Atenção com gate ReLU:** $a_{\text{att}}^{(i)} = \frac{\max(0, \cos(q, k^{(i)}))}{\sum_{i'} \max(0, \cos(q, k^{(i')}))}$, distinguindo o mecanismo do paper de uma atenção tipo softmax convencional.

**3.2.4 Decodificador de Fatores.** AlphaLayer produzindo a distribuição idiossincrática por ativo $(\mu_\alpha, \sigma_\alpha)$ a partir das representações; BetaLayer produzindo a matriz de exposições $\beta \in \mathbb{R}^{N \times K}$ por projeção linear das representações; composição analítica gaussiana fechada para os retornos. A distribuição preditiva é gaussiana com média e desvio em forma fechada, sem amostragem.

> **Fórmula 3.5 — Composição gaussiana analítica:** $\mu_y^{(i)} = \mu_\alpha^{(i)} + \sum_k \beta^{(i,k)} \mu_z^{(k)}$ e $\sigma_y^{(i)} = \sqrt{(\sigma_\alpha^{(i)})^2 + \sum_k (\beta^{(i,k)})^2 (\sigma_z^{(k)})^2}$, fechando a distribuição preditiva sem amostragem.

### 3.3 Treinamento

Subseção única, sem subdivisão. Dois parágrafos curtos.

Primeiro parágrafo apresenta a função objetivo: NLL gaussiana entre retornos observados e a distribuição reconstruída via posterior, mais $\gamma$ vezes a divergência KL entre posterior e prior. Otimização via Adam com learning rate $3 \times 10^{-4}$ e weight decay $10^{-4}$. Batch de uma cross-section por passo, refletindo a natureza multivariada da observação. Early stopping monitorando Rank IC de validação. PyTorch Lightning para organização do treinamento.

Segundo parágrafo: decisões adicionais de estabilização (agendamento do KL, escala relativa entre os termos da loss) descritas no Apêndice A para não sobrecarregar o corpo principal.

> **Fórmula 3.6 — Função objetivo:** $\mathcal{L}(x, y) = -\frac{1}{N} \sum_i \log p_{\phi_{\text{dec}}}(\hat{y}_{\text{rec}}^{(i)} = y^{(i)} | x, z_{\text{post}}) + \gamma \cdot \text{KL}(p_{\phi_{\text{enc}}}(z|x,y) \,\|\, p_{\phi_{\text{pred}}}(z|x))$.

### 3.4 Dados

**3.4.1 Universo de Ativos.** Todos os tickers com pricing válido na Economatica, ponto a ponto, sem restrição ao IBX. O universo médio no período de teste é de aproximadamente 300–500 ativos, contra mais de 3.000 no estudo original — diferença que precisa estar registrada porque condiciona o significado do TopK-Drop com $k = 50$ e as comparações em geral. Mencionar concentração setorial em commodities, energia e financeiro como característica estrutural do mercado brasileiro.

**3.4.2 Período e Divisão da Amostra.** Treinamento, validação e teste explicitados por intervalos de data. O período de teste cobre múltiplos regimes (pandemia de 2020, ciclo de aperto monetário 2021–2022, mudança de governo 2023). Divisão em três subperíodos sem sobreposição.

**3.4.3 Características Utilizadas.**

> **📋 Tabela 1 — Características utilizadas como entrada do modelo.** Vinte características técnicas calculadas a partir de preço e volume, organizadas em categorias (retornos em múltiplos horizontes, volatilidade realizada, volume e turnover, indicadores técnicos, momentos superiores, iliquidez). Colunas: nome da variável, categoria, janela de cálculo. Fórmulas detalhadas no Apêndice B.

**3.4.4 Construção do Universo Point-in-Time.** Subseção dedicada porque é a decisão metodológica que afeta diretamente a credibilidade dos resultados. A inclusão exige janela de $T = 20$ dias consecutivos sem gap calendárico maior que 7 dias (tolerância para feriados). Tickers com suspensões longas ou liquidez intermitente são automaticamente descartados nas datas em que essa condição falha. A composição é reconstruída a cada data, eliminando viés de sobrevivência. A auditoria do procedimento — verificação da presença de tickers que deslistaram dentro do período de treinamento — é descrita no Apêndice C.

**3.4.5 Pré-processamento.** Forward return definido conforme Duan et al. (2022). Filtros de saneamento aplicados para descartar artefatos (preço mínimo de 0,10 BRL, retorno máximo absoluto de 50% em um dia, com justificativa econômica curta). Normalização cross-sectional dos features e dos retornos a cada data. Estatísticas de normalização ancoradas exclusivamente no período de treinamento.

> **Sem figuras adicionais nesta subseção.**

### 3.5 Modelos de Comparação

A escolha dos modelos comparados segue uma progressão deliberada: Momentum (sinal econômico mais simples), Linear/Ridge (limite linear com mesmas características), MLP (modelo neural sem estrutura temporal), GRU (modelo neural com estrutura temporal mas sem componente probabilístico ou estrutura fatorial). Cada baseline isola uma propriedade da arquitetura do FactorVAE.

A ressalva metodológica é explícita: como esses modelos são distintos do FactorVAE em mais de uma dimensão simultaneamente, a comparação não constitui ablação formal. As diferenças observadas são consistentes com a contribuição das propriedades arquiteturais que o FactorVAE adiciona, mas a atribuição causal estrita ficaria a cargo de um estudo de ablação dedicado, fora do escopo deste trabalho.

### 3.6 Procedimentos de Avaliação

**3.6.1 Skill Cross-Sectional.** Rank IC por data e Rank ICIR como razão entre média e desvio ao longo do tempo. As métricas foram apresentadas em 2.4; aqui registra-se apenas a aplicação ao conjunto de teste.

**3.6.2 Estratégia de Portfólio.** TopK-Drop com $k = 50$ ações em carteira e $n = 5$ substituições máximas por dia. A escolha de $k = 50$ segue o paper original e sustenta comparabilidade direta com a literatura, ainda que represente fração maior do universo brasileiro (aproximadamente 25–30% do cross-section típico no teste) do que do A-share chinês (cerca de 1,5%) — ponto que precisa estar registrado. Custo de transação fixo em 10 bps one-way por posição substituída, equivalente a aproximadamente 20 bps round-trip. Benchmark: portfólio igual-ponderado sobre o mesmo universo a cada data.

**3.6.3 Robustez a Ativos Não Vistos.** Teste de holdout-retrain conforme Duan et al. (2022): $m$ tickers removidos do treinamento, modelo retreinado do zero, Rank IC avaliado apenas sobre os $m$ tickers no conjunto de teste. Repetido para múltiplos valores de $m$ e com múltiplas amostras aleatórias.

> **Sem figuras, tabelas ou fórmulas nesta subseção.**

---

## 4 Resultados

Parágrafo de abertura curto: a avaliação é organizada em quatro dimensões — skill cross-sectional, performance ajustada ao risco, comportamento operacional e robustez a ativos não vistos. Nota metodológica explícita: a NLL não é reportada para comparação entre famílias de modelos por se tratar de um limite ELBO no caso do FactorVAE, e não da verdadeira log-verossimilhança marginal — comparação direta seria metodologicamente inválida. Diagnósticos de treinamento ficam restritos ao Apêndice A.

### 4.1 Skill Preditivo Cross-Sectional

Comentário ancorado em números: o FactorVAE lidera em Rank IC, empata com o GRU em Rank ICIR, e a diferença sobre os baselines mais simples é consistente mas não dramática. Sem hedging.

> **📋 Tabela 2 — Qualidade do sinal preditivo.** Linhas: FactorVAE, Momentum, Linear (Ridge), MLP, GRU. Colunas: Rank IC médio, Rank ICIR. FactorVAE destacado.

### 4.2 Performance Ajustada ao Risco

Texto interpreta a tabela em três pontos: o FactorVAE supera os outros sinais em retorno absoluto e ajustado ao risco; o excesso anualizado contra o benchmark igual-ponderado é praticamente nulo após custos; o controle de drawdown está entre os melhores do conjunto. Esse é o resultado a discutir com clareza, não a omitir.

As duas figuras complementam a tabela: a curva de retorno acumulado mostra a trajetória absoluta, e a curva de retorno em excesso mostra o desempenho relativo ao benchmark.

> **📋 Tabela 3 — Performance ajustada ao risco.** Linhas: FactorVAE, EW Market (benchmark), Momentum, Linear (Ridge), MLP, GRU. Colunas: Retorno Anualizado, Retorno em Excesso, Volatilidade, Sharpe, Information Ratio, Calmar, Max Drawdown.

> **🖼️ Figura 3 — Retorno acumulado da estratégia.** Curvas para FactorVAE, baselines neurais (Momentum, Ridge, MLP, GRU) e EW Market. Período de teste completo no eixo x. FactorVAE em destaque (cor primária); benchmark em linha tracejada cinza. Estilo visual alinhado ao plot_style do repositório.

> **🖼️ Figura 4 — Retorno acumulado em excesso vs benchmark.** Mesmo formato da Figura 3, sem a curva do EW Market e com linha horizontal em zero como referência. Destaca o ponto de que o FactorVAE oscila em torno de zero contra o benchmark, mas continua acima dos demais sinais.

### 4.3 Comportamento Operacional da Estratégia

A leitura conecta as duas métricas da tabela ao resultado de portfólio da subseção anterior: o FactorVAE é o único modelo acima de 50% de hit rate e tem turnover moderado em comparação com Ridge, MLP e GRU. A combinação ajuda a explicar por que um ganho modesto em skill se traduz em melhor performance final do que sinais com Rank IC pior ou turnover maior — o sinal é convertido em portfólio com menos atrito.

> **📋 Tabela 4 — Métricas operacionais da estratégia.** Linhas: FactorVAE, Momentum, Linear (Ridge), MLP, GRU. Colunas: Hit Rate, Turnover Médio.

### 4.4 Robustez a Ativos Não Vistos

Apresentação tabular dos resultados do holdout-retrain para os valores de $m$ utilizados, comparados contra o Rank IC de universo completo. Texto interpreta o quanto o desempenho degrada em ativos nunca vistos durante o treinamento e o que isso diz sobre a capacidade de generalização do modelo a IPOs e a ativos pouco líquidos.

> **📋 Tabela 5 — Rank IC sob holdout-retrain.** Linhas: $m = 10$, $m = 50$ (ou outros valores definidos). Colunas: número de trials, Rank IC médio nos held-out, desvio entre trials, Rank IC de universo completo (referência).

---

## 5 Conclusão

Três parágrafos, sem subnumeração formal, no estilo da entrega intermediária.

**Primeiro parágrafo — síntese dos achados.** Retomada da pergunta da introdução em uma frase. Síntese dos achados na ordem em que aparecem nos resultados: o FactorVAE replicado no mercado brasileiro entrega o melhor sinal cross-sectional entre os modelos testados, com vantagem consistente sobre os baselines mais simples e sobreposição parcial com o GRU. A vantagem em skill se traduz em melhor performance de portfólio que os outros sinais, mas o excesso contra o benchmark igual-ponderado é praticamente nulo após custos de transação realistas.

**Segundo parágrafo — limitações.** Quatro limitações concretas. Primeira, o tamanho do dataset (cerca de 5.000 dias de pregão) frente à complexidade do modelo, com efeito provável sobre a estabilidade do treinamento. Segunda, a dependência circular do codificador em retornos contemporâneos como característica do modelo original, que define seu papel como reconstrutor retrospectivo durante o treinamento. Terceira, o custo de transação fixo em 10 bps one-way como aproximação que não captura impacto de mercado em ativos menos líquidos. Quarta, a cobertura de regime do período de teste, que abrange múltiplos ciclos mas não esgota o espaço possível.

**Terceiro parágrafo — extensões futuras.** Específicas, não genéricas. Avaliação em janelas mais longas conforme mais dados ficam disponíveis. Substituição do TopK-Drop por estratégias risk-aware (variantes do tipo TDrisk reportado por Duan et al., utilizando a estimativa $\sigma_y$ que o decodificador entrega em forma fechada). Aplicação do mesmo arcabouço de avaliação a outros mercados emergentes para mapear como a vantagem do FactorVAE depende de características do mercado.

> **Sem figuras, tabelas ou fórmulas nesta seção.**

---

## Apêndices

### Apêndice A — Estabilização do Treinamento

Agendamento linear do termo KL ao longo de 500 passos (`kl_warmup_steps = 500`), com $\gamma$ efetivo iniciando em zero e atingindo o valor da config nesse intervalo.

Decisão sobre o mecanismo de free bits: o threshold por dimensão foi desativado (`kl_free_bits = 0`) após observação de que os valores naturais de KL por fator no regime de operação ficam abaixo de qualquer threshold útil, o que cancelaria gradientes do preditor sem benefício correspondente.

Decisão de tomar média (não soma) sobre $N$ na NLL e sobre $K$ na KL para manter os termos da ELBO em escala compatível dado o cross-section típico.

> **Sem figuras, tabelas ou fórmulas adicionais.**

### Apêndice B — Características Técnicas

> **📋 Tabela B.1 — Detalhamento das vinte características técnicas.** Colunas: nome da variável, fórmula exata, janela, justificativa econômica curta. Inclui retornos em múltiplos horizontes (1, 2, 5, 10, 20 dias), volatilidades realizadas e razões entre elas, indicadores de turnover, desvio do VWAP, RSI, desvios de médias móveis, momentos superiores (skewness e curtose) e medida de iliquidez de Amihud.

### Apêndice C — Diagnósticos Pré-Treino

**Cross-fit Posterior IC.** Teste de validade do codificador como oráculo. Descrição do procedimento: forward sob oráculo na data $t$ (codificador recebe $x_t$ e $y_t$), comparação contra retornos realizados em $t+1$. Se a posterior captura sinal preditivo nos retornos contemporâneos (e não apenas reconstrução trivial), o IC permanece positivo no cross-fit. Resultado obtido reportado em forma textual ou tabular curta.

**Auditoria de viés de sobrevivência.** Distribuição da última data válida por ticker no universo, comparada contra o período de treinamento, com contagem explícita de tickers que deslistaram dentro do treino. Confirma que a construção point-in-time efetivamente preserva tickers que saíram do mercado.

> **Sem figuras adicionais.** Resultados reportados textualmente ou em tabela curta.

### Apêndice D — Hiperparâmetros e Configuração

> **📋 Tabela D.1 — Configuração completa.** Arquivo `config.yaml` em formato tabular: dimensões da arquitetura ($H$, $K$, $M$, $T$, $C$), parâmetros do otimizador (learning rate, weight decay, batch size, número de épocas máximas, paciência do early stopping), seeds, especificação de hardware (GPU utilizada), tempo total de treinamento.

---

## Inventário consolidado

### Figuras no corpo principal

| # | Subseção | Conteúdo |
|---|----------|----------|
| 1 | 2.3 | Diagrama geral da arquitetura do FactorVAE |
| 2 | 3.2 | Diagrama detalhado da arquitetura |
| 3 | 4.2 | Retorno acumulado da estratégia |
| 4 | 4.2 | Retorno acumulado em excesso vs benchmark |

### Tabelas no corpo principal

| # | Subseção | Conteúdo |
|---|----------|----------|
| 1 | 3.4.3 | Características utilizadas como entrada do modelo |
| 2 | 4.1 | Qualidade do sinal preditivo (Rank IC, Rank ICIR) |
| 3 | 4.2 | Performance ajustada ao risco |
| 4 | 4.3 | Métricas operacionais da estratégia |
| 5 | 4.4 | Rank IC sob holdout-retrain |

### Apêndices

| # | Conteúdo |
|---|----------|
| Tabela B.1 | Detalhamento das vinte características técnicas |
| Tabela D.1 | Configuração completa do experimento |

### Fórmulas numeradas

| # | Subseção | Conteúdo |
|---|----------|----------|
| 2.1 | 2.1 | Modelo fatorial geral |
| 2.2 | 2.2 | ELBO |
| 3.1 | 3.1 | Decomposição fatorial dinâmica (operacional) |
| 3.2 | 3.2.2 | Pesos de portfólio (softmax sobre ativos) |
| 3.3 | 3.2.2 | Agregação dos retornos pelos pesos |
| 3.4 | 3.2.3 | Atenção com gate ReLU |
| 3.5 | 3.2.4 | Composição gaussiana analítica |
| 3.6 | 3.3 | Função objetivo |

Apenas oito fórmulas no corpo principal, todas essenciais para a compreensão do modelo. CAPM, APT, Fama-French, Rank IC e Rank ICIR ficam descritos textualmente — o leitor de um TCC em economia já tem familiaridade com esses objetos, e formalizá-los apenas pelo formalismo onera o texto sem ganho de clareza.

---

## Notas sobre o estilo de redação

**Parágrafos de abertura curtos** em cada seção e subseção, situando o leitor antes de entrar no técnico.

**Equações numeradas e referenciadas no texto**, com explicação do que cada termo significa imediatamente após a fórmula. O texto nunca apresenta uma fórmula sem dizer o que cada símbolo representa no parágrafo seguinte.

**Transições explícitas entre subseções** — a frase final de uma subseção prepara a próxima. Esse encadeamento sustenta a leitura linear do trabalho.

**Distância em relação ao paper original.** A descrição do FactorVAE deve ser produzida em redação própria, sem reproduzir a sequência de equações, a notação ou as escolhas de exposição do artigo. O conteúdo é o mesmo (não há como ser diferente, é uma replicação), mas a forma é construída a partir do que o leitor brasileiro de economia precisa para entender o modelo, não a partir de uma tradução do paper.

**Tom.** Capítulos de revisão e metodologia mantêm tom expositivo-analítico; capítulo de resultados mistura descrição e interpretação ancorada em evidência; conclusão é argumentativa-sintética, mais direta e menos técnica. Português acadêmico em todas as seções. Equações em LaTeX. Referências bibliográficas em formato ABNT (compatível com o `abntex2` da entrega intermediária).
