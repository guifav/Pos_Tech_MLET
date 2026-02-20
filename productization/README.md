# Machine Learning Engineering 📈
Camada de produtização para expor, treinar e monitorar modelos LSTM de séries temporais (preço de ações) via FastAPI.

## O que há em `productization`
- `src/app/main.py`: ponto de entrada FastAPI com health checks, `/train` (agendamento assíncrono de treino) e `/infer` (placeholder para predições).
- `src/app/model`: LSTM reutilizável (`LSTM`, `LSTMFactory`, `LSTMParams`) e pipeline de dados com múltiplas estratégias (`DataPipeline`, `DataStrategy` e variantes).
- `src/app/train/model.py`: estratégias de treinamento (`TrainingStrategy`) que combinam pipeline, fábrica do modelo, PyTorch Lightning (`LSTMLightningModule`) e MLflow para rastreio/artefatos.
- `src/app/schemas`: contratos de resposta (`SuccessMessage`, `ErrorMessage`, `RESPONSES`).
- `src/infra/terraform`: infra-as-code para empacotar/deploy (ajuste conforme ambiente).

### Estratégias disponíveis
- Sem feature engineering: `NoProcessingSingleStrategy`, `NoProcessingMultipleStrategy`.
- Faixa diária (High-Low): `RangeSingleStrategy`, `RangeMultipleStrategy`.
- Faixa diária + clustering DBSCAN: `RangeClusterMultipleStrategy`.
- Variante mais profunda: `RangeClusterComplexStrategy`.

### Diagrama de classes (núcleo de treino e dados)
```mermaid
classDiagram
	class FastAPI {
		+POST /train(strategy, params)
		+POST /infer(data)
	}

	class TrainingStrategy {
		<<abstract>>
		+name
		+get_data_pipeline()
		+get_model_factory()
		+get_training_params()
	}
	TrainingStrategy <|-- NoProcessingSimpleStrategy
	TrainingStrategy <|-- NoProcessingSingleStrategy
	TrainingStrategy <|-- NoProcessingMultipleStrategy
	TrainingStrategy <|-- RangeSingleStrategy
	TrainingStrategy <|-- RangeMultipleStrategy
	TrainingStrategy <|-- RangeClusterMultipleStrategy
	TrainingStrategy <|-- RangeClusterComplexStrategy

	class DataStrategy {
		<<abstract>>
		+process(tickers, period, seq_len)
	}
	DataStrategy <|-- NoProcessingSingle
	DataStrategy <|-- NoProcessingMultiple
	DataStrategy <|-- RangeSingle
	DataStrategy <|-- RangeMultiple
	DataStrategy <|-- RangeClusterMultiple

	class DataPipeline {
		+run(tickers, period, seq_len) DataLoader
	}
	class LSTMFactory { +create() LSTM }
	class LSTMParams
	class LSTM
	class LSTMLightningModule
	class TrainerContext { +train() Path }

	FastAPI --> TrainerContext
	TrainerContext --> TrainingStrategy
	TrainingStrategy --> DataPipeline
	TrainingStrategy --> LSTMFactory
	DataPipeline --> DataStrategy
	DataPipeline --> LSTM
	LSTMFactory --> LSTMParams
	LSTMLightningModule --> LSTM
```

### Fluxos reconhecidos
```mermaid
sequenceDiagram
	participant Client
	participant API as FastAPI /train
	participant Strategy as TrainingStrategy
	participant Pipeline as DataPipeline
	participant Trainer as PyTorch Lightning
	participant Store as MLflow + .models

	Client->>API: POST /train?strategy=RangeClusterMultiple
	API->>Strategy: instanciar(strategy, params)
	Strategy->>Pipeline: run(tickers, period, seq_len)
	Pipeline->>Pipeline: yfinance + feature engineering + log de params
	Pipeline-->>API: DataLoader
	API->>Trainer: fit(LSTMLightningModule, loaders)
	Trainer-->>Store: métricas + artefato .pt
	API-->>Client: 202 com caminhos esperados
```

```mermaid
sequenceDiagram
	participant Client
	participant API as FastAPI /infer
	participant Model as LSTM
	participant Store as .models

	Client->>API: POST /infer {payload}
	API->>Store: carregar pesos (planejado)
	API->>Model: forward(data)
	Model-->>Client: prediction (placeholder 0.0)
```

### Como usar
1. Defina `TrainingParams` no corpo do POST e escolha a estratégia via query string (`/train?strategy=RangeMultipleStrategy`).
2. Consulte o diretório `src/app/train/mlruns` para métricas e `src/app/train/.models` para pesos salvos após o término.
3. O endpoint `/infer` ainda é um stub — conecte a carga de pesos e pré-processamento para servir previsões reais.
