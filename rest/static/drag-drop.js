const dropZone = document.getElementById('drag-drop');
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('over');

    const files = e.dataTransfer.files;
    console.log(files)
});
