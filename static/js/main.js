document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.querySelector('.file-input');
    const uploadForm = document.querySelector('form');
    const loadingOverlay = document.querySelector('.loading-overlay');
    const uploadArea = document.querySelector('.upload-area');
    const uploadText = document.querySelector('.upload-text');

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                uploadText.textContent = `Selected: ${file.name}`;
                uploadArea.classList.add('active');
            }
        });
    }

    if (uploadForm) {
        uploadForm.addEventListener('submit', (e) => {
            loadingOverlay.style.display = 'flex';
        });
    }
});
