const API_URL = "http://127.0.0.1:5000/predict";

const imageInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const predictBtn = document.getElementById('predictBtn');
const resultDiv = document.getElementById('result');
const predictedClass = document.getElementById('predictedClass');
const confidence = document.getElementById('confidence');
const probBars = document.getElementById('probBars');

let selectedFile = null;

imageInput.addEventListener('change', (e) => {
  selectedFile = e.target.files[0];
  preview.src = URL.createObjectURL(selectedFile);
  preview.style.display = 'block';
});

predictBtn.addEventListener('click', async () => {
  if (!selectedFile) { alert('Please select an image first'); return; }

  const formData = new FormData();
  formData.append('file', selectedFile);

  predictBtn.textContent = 'Predicting...';
  const res = await fetch(API_URL, { method: 'POST', body: formData });
  const data = await res.json();
  predictBtn.textContent = 'Predict';

  predictedClass.textContent = `Prediction: ${data.predicted_class}`;
  confidence.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;

  probBars.innerHTML = '';
  for (const [cls, prob] of Object.entries(data.all_probabilities)) {
    probBars.innerHTML += `
      <div class="bar-row">
        <span>${cls}</span>
        <div class="bar-bg"><div class="bar-fill" style="width:${prob*100}%"></div></div>
        <span>${(prob*100).toFixed(1)}%</span>
      </div>`;
  }
  resultDiv.style.display = 'block';
});