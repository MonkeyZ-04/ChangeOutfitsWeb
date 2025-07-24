# Change Outfits Web - A.I. Virtual Try-On

This project is a web application for a virtual A.I. try-on. Users can upload a photo of themselves, and the system will use A.I. models to remove the original shirt, leaving only the head and neck. This is then overlaid onto a new, selectable shirt image.

Users can adjust the size and position of their face to fit the new shirt perfectly and can download the final result as an image file.

**Core Technologies:**

  * **Backend:** FastAPI (Python)
  * **Frontend:** HTML, CSS, JavaScript (No Framework)
  * **A.I. Models:**
      * `rembg`: For removing the background from the user's image.
      * `Segment Anything Model (SAM)`: For accurately identifying and masking the clothing to be removed.

-----

## Project Structure

```
/ChangeOutfitsWeb
|
├── backend/
|   ├── main.py             # FastAPI API code
|   ├── requirements.txt      # Python libraries to install
|   └── models/
|       └── sam_vit_b_01ec64.pth # (Will be downloaded automatically)
|
└── frontend/
    ├── index.html          # Main web page
    ├── style.css           # CSS stylesheet
    ├── script.js           # JavaScript logic
    └── images/
        ├── guard_gray.png    # Example shirt image
        └── ... (Other shirt image files)
```

-----

## Setup and Usage

### 1\. Backend Setup (API Server)

This part handles image processing with the A.I. models.

**Steps:**

1.  **Install Python:** Make sure you have Python 3.8 or newer installed.
2.  **Install Libraries:** Open a terminal or command prompt, navigate to the `backend` folder, and run:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Model Download:** The first time you run the server, the `main.py` script will automatically download the `sam_vit_b_01ec64.pth` model file (approx. 300MB) and save it in the `backend/models/` folder.
4.  **Run Server:** Once the installation is complete, run the following command inside the `backend` folder:
    ```bash
    uvicorn main:app --reload
    ```
    The server will start and be accessible at `http://127.0.0.1:8000`.

### 2\. Frontend Usage

This is the user-facing web page.

**Steps:**

1.  **Open the File:** No installation is needed. Simply open the `frontend/index.html` file in any web browser (like Google Chrome or Firefox).
2.  **How to Use:**
      * Upload your photo.
      * Click the "Process Image" button.
      * Wait for the A.I. to process the image.
      * Once the result appears, you can:
          * **Select** different shirts from the dropdown menu.
          * **Adjust** the size and position of the face using the sliders.
          * **Download** the final image as a `.png` file.

-----

## Customization

### Adding More Shirts

You can easily add more shirts for users to choose from:

1.  Place your new shirt image files (PNGs with a transparent neck area are recommended) into the `frontend/images/` folder.
2.  Open the `frontend/script.js` file.
3.  Add a new entry for your shirt to the `shirts` list at the top of the file, following this format:
    ```javascript
    const shirts = [
        // ...existing shirts...
        { name: "The name you want to display", file: "your-image-filename.png" }
    ];
    ```
4.  Save the file and refresh the web page. The new shirt will now be available in the dropdown menu.