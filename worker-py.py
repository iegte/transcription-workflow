"""Mistral Workflow: vergaderaudio transcriberen met Voxtral Mini Transcribe 2.

Bestand plaatsen als: worker.py (root van de repository transcription-workflow)
Start de worker; registreert automatisch de workflow 'meeting-transcription'.
"""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
from datetime import timedelta
from typing import Any

import httpx
import mistralai_workflows as workflows
from mistralai.client import Mistral
from mistralai.client.models import File
from mistralai.workflows import activity_heartbeat
from pydantic import BaseModel, Field


class TranscriptionInput(BaseModel):
    audio_url: str | None = Field(
        default=None,
        description="Publieke URL van het audiobestand (bijv. mp3, wav).",
    )
    audio_base64: str | None = Field(
        default=None,
        description="Audiobestand als base64-string (alternatief voor audio_url).",
    )
    file_name: str = Field(
        default="meeting.mp3",
        description="Bestandsnaam (gebruikt voor content-type bij base64-input).",
    )
    language: str | None = Field(
        default=None,
        description="ISO-taalcode, bijv. 'nl'. Weglaten = automatische taaldetectie.",
    )
    diarize: bool = Field(
        default=True,
        description="Sprekers herkennen (wie zegt wat).",
    )
    timestamps: bool = Field(
        default=True,
        description="Segment-timestamps toevoegen.",
    )
    context_bias: list[str] | None = Field(
        default=None,
        description="Tot 100 woorden/zinnen om namen en vakjargon beter te herkennen.",
    )


class TranscriptionOutput(BaseModel):
    text: str
    language: str | None = None
    segments: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)


@workflows.activity(
    start_to_close_timeout=timedelta(minutes=30),
    heartbeat_timeout=timedelta(seconds=60),
    retry_policy_max_attempts=3,
    retry_policy_backoff_coefficient=2.0,
)
async def transcribe_audio(
    audio_url: str | None,
    audio_base64: str | None,
    file_name: str,
    language: str | None,
    diarize: bool,
    timestamps: bool,
    context_bias: list[str] | None,
) -> TranscriptionOutput:
    """Download het audiobestand en transcribeer het met de Mistral Transcription API."""
    activity_heartbeat({"stage": "downloading"})
    tmp_path = None
    try:
        if audio_url:
            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
                response = await client.get(audio_url)
                response.raise_for_status()
                data = response.content
        elif audio_base64:
            data = base64.b64decode(audio_base64)
        else:
            raise ValueError("Geef audio_url of audio_base64 op.")

        # Tijdelijk bestand: de API verwacht een bestandsobject met een naam.
        suffix = os.path.splitext(file_name)[1] or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        activity_heartbeat({"stage": "transcribing", "bytes": len(data)})

        mistral_client = Mistral()  # gebruikt MISTRAL_SA_TOKEN_PATH / MISTRAL_API_KEY
        with open(tmp_path, "rb") as f:
            response = mistral_client.audio.transcriptions.complete(
                model="voxtral-mini-latest",
                file=File(content=f, file_name=file_name),
                language=language,
                diarize=diarize,
                timestamp_granularities=["segment"] if timestamps else None,
                context_bias=context_bias,
            )

        segments = []
        if timestamps and response.segments:
            for seg in response.segments:
                segments.append(
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "speaker_id": getattr(seg, "speaker_id", None),
                        "text": seg.text,
                    }
                )

        return TranscriptionOutput(
            text=response.text,
            language=getattr(response, "language", None),
            segments=segments,
            usage=response.usage.model_dump() if response.usage else {},
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@workflows.workflow.define(
    name="meeting-transcription",
    workflow_display_name="Meeting Transcription",
    workflow_description=(
        "Transcribeert vergaderaudio naar tekst met Voxtral Mini Transcribe 2, "
        "met diarization en timestamps."
    ),
)
class MeetingTranscriptionWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: TranscriptionInput) -> TranscriptionOutput:
        return await transcribe_audio(
            audio_url=input.audio_url,
            audio_base64=input.audio_base64,
            file_name=input.file_name,
            language=input.language,
            diarize=input.diarize,
            timestamps=input.timestamps,
            context_bias=input.context_bias,
        )


async def main() -> None:
    await workflows.run_worker([MeetingTranscriptionWorkflow])


if __name__ == "__main__":
    asyncio.run(main())