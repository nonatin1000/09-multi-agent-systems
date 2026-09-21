"""Schemas de request/response HTTP (equivalente a ``src/http/chat-schema.ts``)."""

from __future__ import annotations

import re
import uuid

from pydantic import BaseModel, Field, field_validator

from app.graph.router import PRODUCTION_ROUTES

_STRATEGY_ALIASES = {*PRODUCTION_ROUTES, "plan-and-execute"}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: str = Field(min_length=1, alias="userId")
    strategy: str | None = None
    reflect: bool = False
    conversation_id: str | None = Field(default=None, alias="conversationId")
    await_human_approval: bool = Field(default=False, alias="awaitHumanApproval")

    model_config = {"populate_by_name": True}

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, value: str | None) -> str | None:
        if value is not None and value not in _STRATEGY_ALIASES:
            raise ValueError("unknown strategy")
        return value

    @field_validator("conversation_id")
    @classmethod
    def _validate_conversation_id(cls, value: str | None) -> str | None:
        if value is not None:
            uuid.UUID(value)
        return value


class RememberRequest(BaseModel):
    user_id: str = Field(min_length=1, alias="userId")
    fact: str = Field(min_length=1)

    model_config = {"populate_by_name": True}


class ApprovalDecisionBody(BaseModel):
    decision: str = Field(pattern="^(approve|deny)$")
    user_id: str = Field(min_length=1, alias="userId")

    model_config = {"populate_by_name": True}


_SINCE_PATTERN = re.compile(r"^\d+(ms|s|m|h|d)$", re.IGNORECASE)


class StatsQuery(BaseModel):
    """Query ``since`` para GET /stats — ex.: ``24h``, ``7d``, ``30m``."""

    since: str = "24h"

    @field_validator("since")
    @classmethod
    def _validate_since(cls, value: str) -> str:
        if not _SINCE_PATTERN.match(value):
            raise ValueError("since must look like 24h, 7d, 30m")
        return value
