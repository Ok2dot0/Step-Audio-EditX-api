"""
HTTP API server for Step-Audio-EditX model
Provides RESTful endpoints for voice cloning and audio editing functionality
"""
import os
import logging
import tempfile
from datetime import datetime, timezone
from typing import Optional, List

import torch
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Project imports
from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS
from model_loader import ModelSource
from config.edit_config import get_supported_edit_types

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for model instances
tts_engine: Optional[StepAudioTTS] = None
audio_tokenizer: Optional[StepAudioTokenizer] = None

# Pydantic models for API responses
class EditTypeInfo(BaseModel):
    """Model for edit type information"""
    name: str
    options: List[str]


class EditTypesResponse(BaseModel):
    """Response model for edit types endpoint"""
    edit_types: List[EditTypeInfo]


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str
    model_loaded: bool
    timestamp: str


class ErrorResponse(BaseModel):
    """Response model for error messages"""
    error: str
    detail: Optional[str] = None


def create_app(
    model_path: str,
    model_source: str = "auto",
    tokenizer_model_id: str = "dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online",
    tts_model_id: Optional[str] = None,
    quantization: Optional[str] = None,
    torch_dtype_str: str = "bfloat16",
    device_map: str = "cuda"
) -> FastAPI:
    """
    Create and configure the FastAPI application with model loading
    
    Args:
        model_path: Path to the model directory
        model_source: Model source (auto/local/modelscope/huggingface)
        tokenizer_model_id: Tokenizer model ID for online loading
        tts_model_id: TTS model ID for online loading
        quantization: Quantization configuration (int4/int8/awq-4bit)
        torch_dtype_str: PyTorch data type string
        device_map: Device mapping for model loading
    
    Returns:
        Configured FastAPI application
    """
    global tts_engine, audio_tokenizer
    
    app = FastAPI(
        title="Step-Audio-EditX API",
        description="HTTP API for audio editing and zero-shot voice cloning using Step-Audio-EditX model",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Map string arguments to actual types
    source_mapping = {
        "auto": ModelSource.AUTO,
        "local": ModelSource.LOCAL,
        "modelscope": ModelSource.MODELSCOPE,
        "huggingface": ModelSource.HUGGINGFACE
    }
    model_source_enum = source_mapping.get(model_source, ModelSource.AUTO)
    
    # Map torch dtype string to actual torch dtype
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype = dtype_mapping.get(torch_dtype_str, torch.bfloat16)
    
    logger.info(f"Loading models with source: {model_source}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Tokenizer model ID: {tokenizer_model_id}")
    logger.info(f"Torch dtype: {torch_dtype_str}")
    logger.info(f"Device map: {device_map}")
    if tts_model_id:
        logger.info(f"TTS model ID: {tts_model_id}")
    if quantization:
        logger.info(f"🔧 {quantization.upper()} quantization enabled")
    
    # Initialize models
    try:
        # Load StepAudioTokenizer
        audio_tokenizer = StepAudioTokenizer(
            os.path.join(model_path, "Step-Audio-Tokenizer"),
            model_source=model_source_enum,
            funasr_model_id=tokenizer_model_id
        )
        logger.info("✓ StepAudioTokenizer loaded successfully")
        
        # Initialize TTS engine
        tts_engine = StepAudioTTS(
            os.path.join(model_path, "Step-Audio-EditX-AWQ-4bit" if quantization == "awq-4bit" else "Step-Audio-EditX"),
            audio_tokenizer,
            model_source=model_source_enum,
            tts_model_id=tts_model_id,
            quantization_config=quantization,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        logger.info("✓ StepAudioTTS loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        logger.error("Please check your model paths and source configuration.")
        raise RuntimeError(f"Failed to load models: {e}")
    
    # Register routes
    _register_routes(app)
    
    return app


def _register_routes(app: FastAPI):
    """Register all API routes"""
    
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """
        Health check endpoint
        
        Returns the current status of the API and whether models are loaded.
        """
        return HealthResponse(
            status="healthy" if tts_engine is not None else "unhealthy",
            model_loaded=tts_engine is not None,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    @app.get("/api/edit-types", response_model=EditTypesResponse, tags=["Information"])
    async def get_edit_types():
        """
        Get supported edit types and their options
        
        Returns a list of all supported edit types (clone, emotion, style, etc.)
        and their available options/sub-types.
        """
        edit_types = get_supported_edit_types()
        result = [
            EditTypeInfo(name=name, options=options)
            for name, options in edit_types.items()
        ]
        return EditTypesResponse(edit_types=result)
    
    @app.post("/api/clone", tags=["Audio Processing"])
    async def clone_voice(
        background_tasks: BackgroundTasks,
        prompt_audio: UploadFile = File(..., description="Reference audio file for voice cloning"),
        prompt_text: str = Form(..., description="Text content of the reference audio"),
        target_text: str = Form(..., description="Text to synthesize with the cloned voice"),
        output_format: str = Form("wav", description="Output audio format (wav, mp3)")
    ):
        """
        Clone voice from reference audio
        
        Upload a reference audio file along with its text transcription,
        and provide the target text you want to synthesize with the cloned voice.
        
        **Parameters:**
        - `prompt_audio`: Reference audio file (WAV, MP3, etc.)
        - `prompt_text`: Transcription of the reference audio
        - `target_text`: Text to generate speech for
        - `output_format`: Output format (wav or mp3)
        
        **Returns:**
        - Generated audio file with the cloned voice
        """
        if tts_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Validate inputs
        if not prompt_text or prompt_text.strip() == "":
            raise HTTPException(status_code=400, detail="prompt_text cannot be empty")
        if not target_text or target_text.strip() == "":
            raise HTTPException(status_code=400, detail="target_text cannot be empty")
        
        temp_input_path = None
        temp_output_path = None
        
        try:
            # Save uploaded audio to temporary file
            temp_input_path = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ).name
            
            content = await prompt_audio.read()
            with open(temp_input_path, "wb") as f:
                f.write(content)
            
            logger.info(f"Processing clone request: prompt_text='{prompt_text[:50]}...', target_text='{target_text[:50]}...'")
            
            # Perform voice cloning
            output_audio, output_sr = tts_engine.clone(
                temp_input_path, prompt_text, target_text
            )
            
            if output_audio is None or output_sr is None:
                raise HTTPException(status_code=500, detail="Voice cloning failed")
            
            # Convert tensor to numpy if needed
            if isinstance(output_audio, torch.Tensor):
                audio_numpy = output_audio.cpu().numpy().squeeze()
            else:
                audio_numpy = output_audio
            
            # Ensure audio is 2D for soundfile
            if audio_numpy.ndim == 1:
                audio_numpy = audio_numpy.reshape(1, -1)
            
            # Save output to temporary file
            temp_output_path = tempfile.NamedTemporaryFile(
                suffix=f".{output_format}", delete=False
            ).name
            
            # Transpose for soundfile (expects samples x channels)
            sf.write(temp_output_path, audio_numpy.T, output_sr)
            
            logger.info(f"Clone successful, output saved to {temp_output_path}")
            
            # Schedule cleanup of input file
            background_tasks.add_task(os.unlink, temp_input_path)
            
            # Return the audio file
            return FileResponse(
                temp_output_path,
                media_type=f"audio/{output_format}",
                filename=f"cloned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}",
                background=BackgroundTasks([lambda: os.unlink(temp_output_path)])
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Clone failed: {str(e)}")
            # Clean up temp files on error
            if temp_input_path and os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if temp_output_path and os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
            raise HTTPException(status_code=500, detail=f"Clone failed: {str(e)}")
    
    @app.post("/api/edit", tags=["Audio Processing"])
    async def edit_audio(
        background_tasks: BackgroundTasks,
        input_audio: UploadFile = File(..., description="Audio file to edit"),
        audio_text: str = Form("", description="Text content of the input audio (required for most edit types)"),
        edit_type: str = Form(..., description="Type of edit (emotion, style, denoise, vad, paralinguistic, speed)"),
        edit_info: Optional[str] = Form(None, description="Specific edit option (e.g., 'happy', 'whisper', 'faster')"),
        target_text: Optional[str] = Form(None, description="Target text for paralinguistic editing"),
        output_format: str = Form("wav", description="Output audio format (wav, mp3)")
    ):
        """
        Edit audio with various effects
        
        Apply different types of edits to the uploaded audio file.
        
        **Edit Types:**
        - `emotion`: Change emotional tone (happy, sad, angry, etc.)
        - `style`: Change speaking style (whisper, child, serious, etc.)
        - `denoise`: Remove background noise
        - `vad`: Remove silence from audio
        - `paralinguistic`: Add non-verbal sounds ([Laughter], [Breathing], etc.)
        - `speed`: Adjust speaking speed (faster, slower)
        
        **Parameters:**
        - `input_audio`: Audio file to edit
        - `audio_text`: Transcription of the input audio (not required for denoise/vad)
        - `edit_type`: Type of edit to apply
        - `edit_info`: Specific edit option (required for emotion, style, speed)
        - `target_text`: New text with paralinguistic tags (only for paralinguistic type)
        - `output_format`: Output format (wav or mp3)
        
        **Returns:**
        - Edited audio file
        """
        if tts_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Validate edit_type
        supported_types = get_supported_edit_types()
        if edit_type not in supported_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported edit_type: {edit_type}. Supported types: {list(supported_types.keys())}"
            )
        
        # Validate edit_info for types that require it
        if edit_type in ["emotion", "style", "speed"] and not edit_info:
            raise HTTPException(
                status_code=400,
                detail=f"edit_info is required for edit_type '{edit_type}'. Available options: {supported_types[edit_type]}"
            )
        
        if edit_type in ["emotion", "style", "speed"]:
            if supported_types[edit_type] and edit_info not in supported_types[edit_type]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid edit_info '{edit_info}' for edit_type '{edit_type}'. Available options: {supported_types[edit_type]}"
                )
        
        # Validate audio_text for types that require it
        if edit_type not in ["denoise", "vad"] and (not audio_text or audio_text.strip() == ""):
            raise HTTPException(
                status_code=400,
                detail=f"audio_text is required for edit_type '{edit_type}'"
            )
        
        # Validate target_text for paralinguistic
        if edit_type == "paralinguistic" and (not target_text or target_text.strip() == ""):
            raise HTTPException(
                status_code=400,
                detail="target_text is required for paralinguistic editing"
            )
        
        temp_input_path = None
        temp_output_path = None
        
        try:
            # Save uploaded audio to temporary file
            temp_input_path = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ).name
            
            content = await input_audio.read()
            with open(temp_input_path, "wb") as f:
                f.write(content)
            
            logger.info(f"Processing edit request: edit_type='{edit_type}', edit_info='{edit_info}'")
            
            # For paralinguistic, use target_text; otherwise use audio_text
            text_for_edit = target_text if edit_type == "paralinguistic" else audio_text
            
            # Perform audio editing
            output_audio, output_sr = tts_engine.edit(
                temp_input_path,
                audio_text,
                edit_type,
                edit_info,
                text_for_edit
            )
            
            if output_audio is None or output_sr is None:
                raise HTTPException(status_code=500, detail="Audio editing failed")
            
            # Convert tensor to numpy if needed
            if isinstance(output_audio, torch.Tensor):
                audio_numpy = output_audio.cpu().numpy().squeeze()
            else:
                audio_numpy = output_audio
            
            # Ensure audio is 2D for soundfile
            if audio_numpy.ndim == 1:
                audio_numpy = audio_numpy.reshape(1, -1)
            
            # Save output to temporary file
            temp_output_path = tempfile.NamedTemporaryFile(
                suffix=f".{output_format}", delete=False
            ).name
            
            # Transpose for soundfile (expects samples x channels)
            sf.write(temp_output_path, audio_numpy.T, output_sr)
            
            logger.info(f"Edit successful, output saved to {temp_output_path}")
            
            # Schedule cleanup of input file
            background_tasks.add_task(os.unlink, temp_input_path)
            
            # Return the audio file
            return FileResponse(
                temp_output_path,
                media_type=f"audio/{output_format}",
                filename=f"edited_{edit_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}",
                background=BackgroundTasks([lambda: os.unlink(temp_output_path)])
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Edit failed: {str(e)}")
            # Clean up temp files on error
            if temp_input_path and os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if temp_output_path and os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
            raise HTTPException(status_code=500, detail=f"Edit failed: {str(e)}")
    
    @app.post("/api/edit/iterative", tags=["Audio Processing"])
    async def edit_audio_iterative(
        background_tasks: BackgroundTasks,
        input_audio: UploadFile = File(..., description="Audio file to edit"),
        audio_text: str = Form("", description="Text content of the input audio"),
        edit_type: str = Form(..., description="Type of edit"),
        edit_info: Optional[str] = Form(None, description="Specific edit option"),
        target_text: Optional[str] = Form(None, description="Target text for paralinguistic editing"),
        n_iterations: int = Form(1, description="Number of edit iterations (1-5)", ge=1, le=5),
        output_format: str = Form("wav", description="Output audio format")
    ):
        """
        Apply iterative audio editing
        
        For emotion and style editing, applying multiple iterations can enhance the effect.
        This endpoint allows you to specify the number of iterations to apply.
        
        **Parameters:**
        - `input_audio`: Audio file to edit
        - `audio_text`: Transcription of the input audio
        - `edit_type`: Type of edit to apply
        - `edit_info`: Specific edit option
        - `target_text`: New text (only for paralinguistic type)
        - `n_iterations`: Number of iterations to apply (1-5)
        - `output_format`: Output format (wav or mp3)
        
        **Returns:**
        - Final edited audio file after all iterations
        """
        if tts_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Validate edit_type
        supported_types = get_supported_edit_types()
        if edit_type not in supported_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported edit_type: {edit_type}"
            )
        
        # Validate edit_info for types that require it
        if edit_type in ["emotion", "style", "speed"] and not edit_info:
            raise HTTPException(
                status_code=400,
                detail=f"edit_info is required for edit_type '{edit_type}'"
            )
        
        # Validate audio_text for types that require it
        if edit_type not in ["denoise", "vad"] and (not audio_text or audio_text.strip() == ""):
            raise HTTPException(
                status_code=400,
                detail=f"audio_text is required for edit_type '{edit_type}'"
            )
        
        temp_files = []
        
        try:
            # Save uploaded audio to temporary file
            temp_input_path = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ).name
            temp_files.append(temp_input_path)
            
            content = await input_audio.read()
            with open(temp_input_path, "wb") as f:
                f.write(content)
            
            logger.info(f"Processing iterative edit: edit_type='{edit_type}', iterations={n_iterations}")
            
            current_audio_path = temp_input_path
            current_text = audio_text
            
            # Apply iterations
            for i in range(n_iterations):
                logger.info(f"Iteration {i + 1}/{n_iterations}")
                
                text_for_edit = target_text if edit_type == "paralinguistic" else current_text
                
                output_audio, output_sr = tts_engine.edit(
                    current_audio_path,
                    current_text,
                    edit_type,
                    edit_info,
                    text_for_edit
                )
                
                if output_audio is None:
                    raise HTTPException(status_code=500, detail=f"Edit failed at iteration {i + 1}")
                
                # Convert and save intermediate result
                if isinstance(output_audio, torch.Tensor):
                    audio_numpy = output_audio.cpu().numpy().squeeze()
                else:
                    audio_numpy = output_audio
                
                if audio_numpy.ndim == 1:
                    audio_numpy = audio_numpy.reshape(1, -1)
                
                # Save intermediate result for next iteration
                if i < n_iterations - 1:
                    intermediate_path = tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ).name
                    temp_files.append(intermediate_path)
                    sf.write(intermediate_path, audio_numpy.T, output_sr)
                    current_audio_path = intermediate_path
                    
                    # Update text if paralinguistic
                    if edit_type == "paralinguistic" and target_text:
                        current_text = target_text
            
            # Save final output
            temp_output_path = tempfile.NamedTemporaryFile(
                suffix=f".{output_format}", delete=False
            ).name
            sf.write(temp_output_path, audio_numpy.T, output_sr)
            
            logger.info(f"Iterative edit successful after {n_iterations} iterations")
            
            # Schedule cleanup
            for f in temp_files:
                background_tasks.add_task(os.unlink, f)
            
            return FileResponse(
                temp_output_path,
                media_type=f"audio/{output_format}",
                filename=f"edited_{edit_type}_iter{n_iterations}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}",
                background=BackgroundTasks([lambda: os.unlink(temp_output_path)])
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Iterative edit failed: {str(e)}")
            for f in temp_files:
                if os.path.exists(f):
                    os.unlink(f)
            raise HTTPException(status_code=500, detail=f"Iterative edit failed: {str(e)}")


# Main entry point for running the API server
if __name__ == "__main__":
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Step-Audio-EditX HTTP API Server")
    parser.add_argument("--model-path", type=str, required=True, help="Model path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--model-source",
        type=str,
        default="auto",
        choices=["auto", "local", "modelscope", "huggingface"],
        help="Model source"
    )
    parser.add_argument(
        "--tokenizer-model-id",
        type=str,
        default="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online",
        help="Tokenizer model ID"
    )
    parser.add_argument(
        "--tts-model-id",
        type=str,
        default=None,
        help="TTS model ID"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["int4", "int8", "awq-4bit"],
        help="Quantization configuration"
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="PyTorch data type"
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="cuda",
        help="Device mapping"
    )
    
    args = parser.parse_args()
    
    # Create the app
    app = create_app(
        model_path=args.model_path,
        model_source=args.model_source,
        tokenizer_model_id=args.tokenizer_model_id,
        tts_model_id=args.tts_model_id,
        quantization=args.quantization,
        torch_dtype_str=args.torch_dtype,
        device_map=args.device_map
    )
    
    # Run the server
    uvicorn.run(app, host=args.host, port=args.port)
