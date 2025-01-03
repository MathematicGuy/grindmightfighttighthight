// Remove the form submission event listener
document.getElementById('readVietnameseTextButton').addEventListener('click', readVietnameseText);

async function uploadImage() {
    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];
    const loadingElement = document.getElementById('loading');
    const resultsElement = document.getElementById('results');
    const readTextButton = document.getElementById('readVietnameseTextButton');

    if (!file) {
        alert("No image selected. Please choose an image to upload.");
        return;
    }

    const originalFileName = file.name;
    const jsonFileName = 'ocr_' + originalFileName.substring(0, originalFileName.lastIndexOf('.')) + '.json';
    sessionStorage.setItem('fileName', jsonFileName); // Save the JSON file name
    readTextButton.disabled = false; // Enable the button after successful upload

    const formData = new FormData();
    formData.append('file', file);

    loadingElement.style.display = 'block';
    resultsElement.style.display = 'none'; // Hide previous results

    try {
        const response = await fetch("http://127.0.0.1:8000/detect/", {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            alert(`Error: ${error.detail}`);
            loadingElement.style.display = 'none';
            return;
        }

        console.log("Original File Name:", originalFileName);
        console.log("JSON File Name:", jsonFileName);
        const ocr_text = await fetch(`../validation/detect_text/${jsonFileName}`); // Use the constructed JSON file name

        if (!ocr_text.ok) {
            alert("Error fetching Vietnamese text.");
            loadingElement.style.display = 'none';
            return;
        }
        
        const data = await ocr_text.json();
        document.getElementById('ocrText').textContent = JSON.stringify(data, null, 2);
        document.getElementById('results').style.display = 'block';
    } catch (error) {
        console.error("Error during upload:", error);
    } finally {
        loadingElement.style.display = 'none'; // Hide loading after success or error
        readTextButton.disabled = false; // Enable the button after successful upload
    }
}

async function readVietnameseText() {
    const fileName = sessionStorage.getItem('fileName'); // Retrieve the JSON file name from sessionStorage

    if (!fileName) {
        alert("No file has been uploaded yet.");
        return;
    }

    try {
        const ocr_text = await fetch(`../validation/detect_text/${fileName}`); // Use the retrieved JSON file name

        if (!ocr_text.ok) {
            throw new Error("Network response was not ok");
        }

        const data = await ocr_text.json();
        document.getElementById('ocrText').textContent = JSON.stringify(data, null, 2);
        document.getElementById('results').style.display = 'block';
    } catch (error) {
        console.error("Error fetching or parsing OCR text:", error);
    }
}

// Initialize the button state on window load
window.onload = function() {
    const storedFileName = sessionStorage.getItem('fileName');
    const readTextButton = document.getElementById('readVietnameseTextButton');
    if (storedFileName) {
        console.log("JSON File name found in sessionStorage:", storedFileName);
        readTextButton.disabled = false;
    } else {
        readTextButton.disabled = true; // Disable initially if no file
    }
};