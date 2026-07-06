import io
import logging
from PIL import Image
from fastapi import HTTPException, UploadFile
from backend.config import Settings

logger = logging.getLogger(__name__)

def validate_uploaded_image(file: UploadFile, content: bytes, settings: Settings) -> None:
    """
    Validates an uploaded image file:
    1. Checks if the file is empty.
    2. Checks if the file size exceeds configured limits.
    3. Validates MIME type against allowed types.
    4. Validates file signature (magic bytes) to reject renamed executables or unsupported formats.
    5. Verifies image integrity and structure using Pillow.
    
    Raises HTTPException (400, 413, 415) for validation failures.
    """
    # 1. Reject empty files
    if not content or len(content) == 0:
        logger.warning(
            "Rejected upload with empty file content.",
            extra={"event": "validation_failed", "detail": "empty_file"}
        )
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 2. Reject files exceeding max size
    if len(content) > settings.max_upload_size:
        max_mb = settings.max_upload_size / (1024 * 1024)
        logger.warning(
            f"Rejected upload exceeding maximum size limit: {len(content)} bytes",
            extra={"event": "validation_failed", "detail": "file_too_large"}
        )
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds the maximum limit of {max_mb:.1f}MB."
        )

    # 3. Validate MIME type
    if file.content_type not in settings.allowed_mime_types:
        logger.warning(
            f"Rejected upload with unsupported MIME type: {file.content_type}",
            extra={"event": "validation_failed", "detail": "unsupported_mime_type"}
        )
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: '{file.content_type}'. Allowed types: {', '.join(settings.allowed_mime_types)}"
        )

    # 4. Reject renamed executable files (MZ, ELF, Mach-O, Shebang/script)
    # Check for PE/COFF Windows Executables (MZ)
    if content.startswith(b"MZ"):
        logger.warning(
            "Rejected upload containing Windows executable signature.",
            extra={"event": "validation_failed", "detail": "executable_signature"}
        )
        raise HTTPException(status_code=400, detail="Renamed executable files are not allowed.")

    # Check for ELF Linux Executables
    if content.startswith(b"\x7fELF"):
        logger.warning(
            "Rejected upload containing ELF executable signature.",
            extra={"event": "validation_failed", "detail": "executable_signature"}
        )
        raise HTTPException(status_code=400, detail="Renamed executable files are not allowed.")

    # Check for Shebang script files
    if content.startswith(b"#!"):
        logger.warning(
            "Rejected upload containing script shebang signature.",
            extra={"event": "validation_failed", "detail": "script_signature"}
        )
        raise HTTPException(status_code=400, detail="Renamed executable files are not allowed.")

    # Check for macOS Mach-O/Java Class signatures
    mach_o_headers = [
        b"\xca\xfe\xba\xbe",  # Mach-O Fat Binary / Java Class
        b"\xfe\xed\xfa\xce",  # Mach-O 32-bit
        b"\xfe\xed\xfa\xcf",  # Mach-O 64-bit
        b"\xce\xfa\xed\xfe",  # Mach-O 32-bit reverse
        b"\xcf\xfa\xed\xfe"   # Mach-O 64-bit reverse
    ]
    for header in mach_o_headers:
        if content.startswith(header):
            logger.warning(
                "Rejected upload containing Mach-O/Java executable signature.",
                extra={"event": "validation_failed", "detail": "executable_signature"}
            )
            raise HTTPException(status_code=400, detail="Renamed executable files are not allowed.")

    # 5. Validate file signature (magic bytes) for allowed image types
    is_signature_valid = False
    if file.content_type == "image/jpeg":
        is_signature_valid = content.startswith(b"\xff\xd8")
    elif file.content_type == "image/png":
        is_signature_valid = content.startswith(b"\x89PNG\r\n\x1a\n")
    elif file.content_type == "image/webp":
        is_signature_valid = (len(content) >= 12 and content[0:4] == b"RIFF" and content[8:12] == b"WEBP")

    if not is_signature_valid:
        logger.warning(
            f"File signature (magic bytes) does not match MIME type: {file.content_type}",
            extra={"event": "validation_failed", "detail": "signature_mismatch"}
        )
        raise HTTPException(
            status_code=400,
            detail="File signature (magic bytes) does not match the MIME type or format is unsupported."
        )

    # 6. Verify image using Pillow before inference
    try:
        # Perform structural validation
        with Image.open(io.BytesIO(content)) as img:
            img.verify()

        # Perform loading validation to catch corrupted pixel data
        with Image.open(io.BytesIO(content)) as img:
            img.load()
            
            # Map Pillow format to standard MIME types
            pillow_format_to_mime = {
                "JPEG": "image/jpeg",
                "PNG": "image/png",
                "WEBP": "image/webp"
            }
            detected_mime = pillow_format_to_mime.get(img.format)
            if detected_mime != file.content_type:
                logger.warning(
                    f"Pillow detected format {img.format} ({detected_mime}) does not match MIME type: {file.content_type}",
                    extra={"event": "validation_failed", "detail": "pillow_format_mismatch"}
                )
                raise HTTPException(
                    status_code=400,
                    detail="File format detected by parser does not match the declared MIME type."
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(
            f"Pillow validation failed for uploaded image: {e}",
            extra={"event": "validation_failed", "detail": "corrupted_image"}
        )
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or corrupted image: {str(e)}"
        )
