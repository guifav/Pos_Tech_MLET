"""
Modulo para previsão de preços de ações
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __app__, __author__, __version__, logger
from app.schemas import RESPONSES, ErrorMessage

from app import train


tags_metadata: list[dict] = [
    {
        "name": "Treinamento",
        "description": """
    Endpoints para o treinamento e a otimização de um novo modelo.
        """,
    },
    {
        "name": "Inferencia",
        "description": """
    Endpoints de inferência do modelo.
        """,
    },
    {
        "name": "Atualização",
        "description": """
    Endpoints para a atualização do modelo, como fine tunning, prunning e quantization.
        """,
    },
    {
        "name": "Monitoramento",
        "description": """
    Endpoints de monitoramento do modelos.
        """,
    },
    {
        "name": "Configuração",
        "description": """
    Endpoints para configuração do Serviço.
        """,
    },
]

description: str = """

"""


app: FastAPI = FastAPI(
    title=__app__,
    version=__version__,
    description=description,
    openapi_tags=tags_metadata,
    openapi_url="/api/v1/openapi.json",
    responses=RESPONSES,  # type: ignore
    swagger_ui_oauth2_redirect_url='/oauth2-redirect',
    swagger_ui_init_oauth={
        'usePkceWithAuthorizationCodeGrant': True,
        'clientId': f'{os.getenv("MSFT_CLIENT_ID", "")}',
    },
    docs_url=None,
    redoc_url=None
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError  # pylint: disable=unused-argument
) -> JSONResponse:
    """
    validation_exception_handler Exception handler for validations.

    Args:
        request (Request): the request from the api
        exc (RequestValidationError): the validation raised by the process

    Returns:
        JSONResponse: A json encoded response with the validation errors.
    """

    response_body: ErrorMessage = ErrorMessage(
        success=False,
        type="Validation Error",
        title="Your request parameters didn't validate.",
        detail={"invalid-params": list(exc.errors())},
    )

    logger.error(
        f"Validation error: {exc.errors()}",
        extra={
            "request": {
                "method": request.method,
                "url": request.url,
                "headers": request.headers,
                "body": await request.json(),
            }
        },
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(response_body),
    )


@app.exception_handler(ResponseValidationError)
async def response_exception_handler(
    request: Request, exc: ResponseValidationError  # pylint: disable=unused-argument
) -> JSONResponse:
    """
    response_exception_handler Exception handler for response validations.

    Args:
        request (Request): the request from the api
        exc (RequestValidationError): the validation raised by the process

    Returns:
        JSONResponse: A json encoded response with the validation errors.
    """

    response_body: ErrorMessage = ErrorMessage(
        success=False,
        type="Response Error",
        title="Found Errors on processing your requests.",
        detail={"invalid-params": list(exc.errors())},
    )

    logger.error(
        f"Response validation error: {exc.errors()}",
        extra={
            "request": {
                "method": request.method,
                "url": request.url,
                "headers": request.headers,
                "body": await request.json(),
            }
        },
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(response_body),
    )


@app.get("/", tags=["Configuração"], summary="Health Check Endpoint")
async def health_check() -> dict[str, str]:
    """
    health_check Health check endpoint to verify if the service is running.

    Returns:
        dict[str, str]: A simple dictionary indicating the service is healthy.
    """
    return {"status": "healthy"}


@app.get("/ready", tags=["Configuração"], summary="Readiness Check Endpoint")
async def readiness_check() -> dict[str, str]:
    """
    readiness_check Readiness check endpoint to verify if the service is ready to
    accept requests.

    Returns:
        dict[str, str]: A simple dictionary indicating the service is ready.
    """
    return {"status": "ready"}


@app.get("/startup", tags=["Configuração"], summary="Startup Check Endpoint")
async def startup_check() -> dict[str, str]:
    """
    startup_check Startup check endpoint to verify if the service has started
    successfully.

    Returns:
        dict[str, str]: A simple dictionary indicating the service has started.
    """
    return {"status": "started"}


@app.post("/train", tags=["Treinamento"], summary="Train a new LSTM model")
async def train_model(strategy: str, params: train.TrainingParams) -> dict[str, Path]:
    """
    train_model Endpoint to initiate the training of a new LSTM model.

    Returns:
        dict[str, str]: A simple dictionary indicating the training has started.
    """
    strategy_class = getattr(train, strategy)(params)
    path = train.TrainerContext(strategy_class).train()
    return {"model_path": path}


@app.post("/infer", tags=["Inferencia"], summary="Make a prediction using the LSTM model")
async def infer_model(data: dict) -> dict[str, float]:
    """
    infer_model Endpoint to make a prediction using the trained LSTM model.

    Returns:
        dict[str, float]: A dictionary containing the prediction result.
    """
    # Placeholder for inference logic
    prediction_result = 0.0  # Replace with actual prediction logic
    return {"prediction": prediction_result}
