import asyncio
import websockets
from faster_whisper import WhisperModel
import numpy as np
import logging
import json
import signal

# Configure structured logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000
CHANNELS = 1
BUFFER_DURATION = 3  # seconds
BUFFER_SIZE = SAMPLE_RATE * CHANNELS * BUFFER_DURATION
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
MAX_INT16 = 32768.0

class TranscriptionServer:
    def __init__(self):
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize Whisper model with error handling."""
        try:
            logger.info("Initializing Whisper model...")
            self.model = WhisperModel(
                "small.en",
                compute_type="int8",
                cpu_threads=8,
                device="cpu"
            )
            # Warm up the model with a small silent sample
            warmup_audio = np.zeros(16000, dtype=np.float32)
            _ = self.model.transcribe(warmup_audio, beam_size=1)
            logger.info("Model initialized and warmed up")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise

    async def transcribe_audio(self, audio_np):
        """Transcribe audio numpy array."""
        try:
            segments, info = self.model.transcribe(
                audio_np,
                beam_size=5,
                vad_filter=True,
                no_speech_threshold=0.3,
                language="en",
                task="transcribe"
            )
            return " ".join(segment.text for segment in segments), None
        except Exception as e:
            return None, str(e)

    async def handle_client(self, websocket, path=None):
        """Handle incoming WebSocket connections."""
        client_ip = websocket.remote_address[0]
        logger.info(f"New connection from {client_ip}")

        audio_buffer = bytearray()

        try:
            async for message in websocket:
                audio_buffer.extend(message)

                # Process complete chunks
                while len(audio_buffer) >= BUFFER_SIZE:
                    chunk = audio_buffer[:BUFFER_SIZE]
                    del audio_buffer[:BUFFER_SIZE]  # More efficient than slicing

                    # Convert to numpy array
                    try:
                        audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / MAX_INT16
                    except Exception as e:
                        logger.error(f"Audio conversion error for {client_ip}: {e}")
                        await websocket.send("ERROR: Invalid audio format")
                        continue

                    # Transcribe
                    transcript, error = await self.transcribe_audio(audio_np)
                    if error:
                        logger.error(f"Transcription error for {client_ip}: {error}")
                        await websocket.send(f"ERROR: {error}")
                        continue

                    if transcript:
                        print(f"{transcript}")
                        await websocket.send(json.dumps({
                            "type": "transcribe",
                            "content": transcript
                        }))
                    else:
                        logger.debug(f"No speech detected for {client_ip}")
                        await websocket.send("[No speech detected]")

        except websockets.ConnectionClosed:
            logger.info(f"Client disconnected: {client_ip}")
        except Exception as e:
            logger.error(f"Error with {client_ip}: {e}", exc_info=True)
        finally:
            await websocket.close()

    async def shutdown(self):
        """Clean up resources."""
        if self.model:
            logger.info("Cleaning up Whisper model")
            del self.model
            self.model = None

async def shutdown_signal(server, transcription_server, signal):
    """Handle shutdown signals."""
    logger.info(f"Received {signal.name}, shutting down...")
    server.close()
    await server.wait_closed()
    await transcription_server.shutdown()
    logger.info("Server shutdown complete")

async def main():
    transcription_server = TranscriptionServer()

    server = await websockets.serve(
        transcription_server.handle_client,
        "0.0.0.0",
        8765
    )
    logger.info("Server started on ws://0.0.0.0:8765")

    # Setup signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda sig=sig: asyncio.create_task(
                shutdown_signal(server, transcription_server, sig)
            )
        )

    try:
        await server.wait_closed()
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Server stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise
