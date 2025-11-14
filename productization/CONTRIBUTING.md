# Guia de Contribuição

Este repositório é privado para a equipe interna e serve como um **micro-framework para a produtização de modelos LSTM**. As contribuições devem manter coerência arquitetural, qualidade de código e rastreabilidade. Leia este documento antes de abrir qualquer Issue ou Pull Request.

## Sumário

## Escopo e Código de Conduta
Embora privado, seguimos princípios de respeito e colaboração. Aplicamos o [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Qualquer comportamento inadequado deve ser reportado internamente ao responsável técnico do projeto.

 
 
 
## Regras Arquitetônicas e de Contribuição
As regras abaixo refletem decisões consolidadas para garantir consistência em ambiente de desenvolvimento, treinamento e deploy.

 
 
 
### 1. Ambiente VS Code
1.1 Mantenha o ambiente limpo. Somente é permitido commitar arquivos de configuração de debug em `.vscode/launch.json`.
1.2 É proibido versionar arquivos locais que alterem variáveis de ambiente, tarefas personalizadas ou extensões específicas (ex.: `settings.json`, variáveis com segredos ou configs que mudem comportamento de execução individual).
1.3 Não introduza dependências que não estejam descritas em `pyproject.toml`.

 
 
 
### 2. Stack Tecnológica e Qualidade (ADR 001)
2.1 Linguagem: **Python >= 3.13**. Código deve ser compatível com essa versão mínima.
2.2 Redes neurais: **PyTorch** com **PyTorch Lightning** para orquestração de treinamento e **Ray** para processamento distribuído.
2.3 Observabilidade e rastreio de experimentos: **MLflow** (rastreamento de parâmetros, métricas, artefatos e versionamento de modelos).
2.4 Testes: **PyTest** para unidade e integração. Todo novo módulo deve ter cobertura mínima (happy path + 1 edge case).
2.5 Qualidade: **Pylint** para linter e **Black** para formatação. Commits devem estar formatados e sem erros de lint (exceto justificativas documentadas em comentários `# noqa` pontuais).
2.6 API: **FastAPI** para exposição de endpoints de inferência e metadados do modelo.
2.7 Não adicionar bibliotecas duplicadas que resolvam o mesmo problema (ex.: evitar múltiplos frameworks de serving).

 
 
 
### 3. Containerização e Orquestração
3.1 Utilizamos **Docker** para empacotamento e **Kubernetes** para execução.
3.2 É proibido alterar ou remover endpoints padronizados de `liveness`, `readiness` e `startup` probes definidos no setup do repositório (manter contratos para monitoramento e autoscaling).
3.3 Variáveis sensíveis (tokens, chaves) nunca são commitadas; usar mecanismos de Secret/ConfigMap no cluster.

 
 
 
### 4. GitHub Actions e CI/CD
4.1 Workflows devem ser adicionados em branch separada e só depois integrados via Pull Request.
4.2 Pipelines iniciais (lint, testes, build de imagem, smoke inferência) não devem bloquear desenvolvimento experimental até estabilização, mas devem rodar em PR.
4.3 Não inserir steps que alterem estado de branches de desenvolvimento de forma automática (ex.: force push). Deploy automatizado só após aprovação explícita.

 
 
 
### 5. Pull Requests
5.1 Todo PR deve referenciar uma Issue (feature, bug ou melhoria técnica).
5.2 Incluir descrição clara: objetivo, abordagem, impacto arquitetural e links para artefatos (MLflow run, ADR relevante).
5.3 Adicionar checklist: testes passam, lint ok, formatação aplicada, documentação atualizada (se aplicável), endpoints preservados.
5.4 Tamanho: preferir PRs pequenos e incrementais. Se grande, dividir em subtarefas e PRs encadeados.
5.5 Revisões: mínimo 1 revisor técnico + 1 revisor de ML (quando altera lógica de treinamento/inferência).

 
 
 
### 6. Estrutura de Dados e Modelagem
6.1 Inputs para LSTM devem estar encapsulados em schemas (`pydantic`) definidos em `src/app/schemas`.
6.2 Funções de pré-processamento padronizadas em módulo central (`data.py` ou outro definido por ADR futura).
6.3 Parametrizações de treinamento (hyperparameters) mantidas versionadas e rastreadas em MLflow.

 
 
 
### 7. Segurança e Segredos
7.1 Nunca commit secrets. Utilize variáveis de ambiente injetadas no runtime (Docker/Kubernetes).
7.2 Qualquer uso de credenciais para serviços externos (storage, tracking) deve estar documentado.

 
 
 
## Feature Requests
Para solicitar nova funcionalidade:
1. Abrir Issue com o rótulo `feature`.
2. Descrever: contexto, motivação, definição de pronto (DoD), impactos arquiteturais e métricas desejadas.
3. Indicar se requer novos endpoints, ajustes em pipeline de treinamento ou mudanças na estrutura de dados.
4. Se impactar ADR existente, propor atualização ou nova ADR numerada (ex.: ADR 002).

 
 
 
## Issues e Bugs
Ao reportar um bug, inclua:
Uso de rótulos: `bug`, `infra`, `data`, `performance`, `documentation` para triagem eficiente.

 
 
 
## Guia de Pull Requests
Siga este fluxo:
1. Criar branch: `feature/<descricao-curta>` ou `fix/<descricao-curta>`.
2. Implementar mudanças mantendo padrão de código e ferramentas.
3. Executar localmente:
   ```bash
   black .
   pylint src || true  # Corrigir erros críticos
   pytest -q
   ```
4. Atualizar documentação se alterar comportamento público (README, schemas, exemplos de uso da API).
5. Abrir PR relacionando Issue. Preencher checklist e solicitar revisão.
6. Após aprovação, realizar merge (fast-forward ou squash conforme política definida pelo maintainers).

 
 
 
## Estilo de Código e Qualidade

 
 
 
## Testes, Observabilidade e Modelos

 
 
 
## Branching e GitHub Actions

### Estratégia: Trunk-Based Development

Adotamos **trunk-based development** para maximizar fluxo contínuo, reduzir divergências e acelerar feedback. O "trunk" é a branch `main` (estável). Não mantemos branches long-lived (ex.: `develop`) salvo decisão explícita via ADR futura.

#### Conceitos Centrais

- **Commits pequenos e frequentes**: evitar grandes blocos difíceis de revisar e integrar.
- **Branches curtos**: vida útil ideal < 1 dia; excepcionalmente até 2–3 dias se for uma feature maior, dividida em commits claros.
- **Integração contínua**: cada PR dispara pipeline (lint, testes, build container, smoke de inferência) mantendo `main` saudável.
- **Feature flags**: funcionalidades incompletas protegidas por flag até estarem prontas.

#### Criação de Branches

- Padrões: `feat/<slug>` nova funcionalidade; `fix/<slug>` correção; `exp/<slug>` experimento; `chore/<slug>` manutenção; `hotfix/<slug>` correção urgente.
- Slug em kebab-case curto: ex. `feat/lstm-batch-normalization`.
- Hotfix nasce sempre de `main`.

#### Workflow Diário

1. Atualizar base: `git fetch origin && git switch main && git pull --ff-only`.
2. Criar branch: `git switch -c feat/<slug>`.
3. Commits atômicos (mensagem: contexto + ação) ex.: `feat: adiciona normalizacao batch no pre-processamento`.
4. Rebase frequente: `git fetch origin && git rebase origin/main`.
5. Rodar Black, Pylint e PyTest antes do PR.

#### Política de Merge

- Preferir **Squash merge** (histórico limpo + referência à Issue).
- Usar merge preservando commits (`--no-ff`) somente com justificativa (ex.: série de medições de performance).
- Evitar merges grandes e tardios; dividir em slices pequenas.

#### Revisão

- Mínimo: 1 revisor técnico + 1 de ML quando afeta lógica de modelo.
- PR deve incluir motivação, abordagem, impacto, riscos, plano de rollback.
- Não aprovar PR com falhas de lint/testes.

#### Hotfixes

- Branch `hotfix/<issue>` direto de `main`.
- Apenas correção mínima + teste de regressão.
- Merge rápido (squash) e criação de tag se afetar release.

#### Tags e Releases

- SemVer: `vMAJOR.MINOR.PATCH`.
- Patch: correções; Minor: novas features retrocompatíveis; Major: mudanças de contrato (exigem ADR e comunicado).
- Tag após pipeline verde em `main`.

#### Rollback Simples

- Squash: `git revert <hash>`.
- Feature flag: desativar flag e limpar código em novo PR.

#### GitHub Actions

- Sem deploy automático direto em `main` sem aprovação humana (gate manual).
- Checks mínimos: formatação, lint, testes, build container, smoke de inferência.
- Workflows novos via PR dedicado; reaproveitar composite actions quando recorrente.

#### Benefícios Esperados

- Menos conflitos.
- Histórico legível por feature.
- Feedback rápido sobre regressões.
- Rollback simples.

Exceções devem ser documentadas em Issue e, se recorrentes, formalizadas em ADR.

 
 
 
## ADR 001

Registra a decisão pela stack: Python 3.13+, PyTorch + Lightning, Ray, MLflow, PyTest, Pylint, Black, FastAPI. Novas decisões devem seguir numeração sequencial (`ADR 002`, etc.) e manter formato padronizado (Contexto, Decisão, Consequências, Alternativas). Alterações que contrariem ADR exigem revisão formal.

 
 
 
## Suporte e Dúvidas

Para dúvidas técnicas use Issues com rótulo `question` ou canal interno acordado. Evite comunicação fora dos meios para garantir rastreabilidade.

Obrigado por manter a qualidade e evolução consistente do micro-framework de LSTM! Contribua de forma incremental e documentada.
