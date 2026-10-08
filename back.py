import io
import numpy as np
import cv2
import img2pdf
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import fitz  # PyMuPDF


app = FastAPI(title="Image & PDF Processing API")

# Enable CORS for frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/convert_image_to_pdf")
async def convert_image_file_to_pdf(file: UploadFile = File(...)):
    """Uploads a local image file and converts it directly into a PDF download."""
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Please upload a valid image file (.jpg, .jpeg, .png)")
    
    try:
        image_bytes = await file.read()
        pdf_bytes = img2pdf.convert(image_bytes)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=converted.pdf"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion error: {str(e)}")

@app.post("/convert_pdf_to_image")
async def convert_pdf_file_to_image(file: UploadFile = File(...)):
    """Uploads a local PDF file and extracts its first page as a JPEG image."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a valid PDF file (.pdf)")

    try:
        pdf_bytes = await file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        if doc.page_count == 0:
            raise HTTPException(status_code=400, detail="The document contains no readable pages")

        page = doc.load_page(0)  # Load the first page
        pix = page.get_pixmap(dpi=300)  # Render page to image

        img_buffer = io.BytesIO(pix.tobytes("jpeg"))
        img_buffer.seek(0)

        return StreamingResponse(img_buffer, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server extraction error: {str(e)}")

@app.post("/image_cleaning")
async def image_cleaning(file: UploadFile = File(...)):
    """Uploads a local JPG image file, sharpens it, and returns the result as a PDF download."""
    if not file.filename.lower().endswith((".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Please upload a valid JPG/JPEG image")
    
    try:
        file_bytes = await file.read()
        nparray = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparray, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file data")
        
        # Enhanced Sharpening matrix processing
        gaussian_blur = cv2.GaussianBlur(img, (5, 5), 1.0)
        img_sharp = cv2.addWeighted(img, 1.6, gaussian_blur, -0.6, 0) 
        
        _, encode = cv2.imencode(".jpg", img_sharp) 
        clean_jpg_bytes = encode.tobytes()
        pdf_bytes = img2pdf.convert(clean_jpg_bytes)
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=cleaned.pdf"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")
