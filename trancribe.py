import whisper

def trancribe(file_path: str) -> str:
    """
    Load a Whisper model and transcribe an audio file to text.

    Args:
        file_path: Path to the audio file to transcribe.

    Returns:
        Transcribed text string from the audio.
    """
    # Choose model size: options include "tiny", "base", "small", "medium", "large"
    # "medium" offers a balance between speed and accuracy
    model = whisper.load_model("medium")

    # Perform transcription on the provided file path
    whisper_result = model.transcribe(file_path)

    # Extract and return only the textual transcription
    return whisper_result.get("text", "")  # Return empty string if 'text' key is missing
