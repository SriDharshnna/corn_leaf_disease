const API_URL = window.location.origin + "/predict";

const imageInput = document.getElementById('imageInput');
const dropzoneEmpty = document.getElementById('dropzoneEmpty');
const scanStage = document.getElementById('scanStage');
const preview = document.getElementById('preview');
const scanLine = document.getElementById('scanLine');
const predictBtn = document.getElementById('predictBtn');

const reportEmpty = document.getElementById('reportEmpty');
const reportBody = document.getElementById('reportBody');
const verdictClass = document.getElementById('verdictClass');
const verdictConf = document.getElementById('verdictConf');
const warningBanner = document.getElementById('warningBanner');
const meters = document.getElementById('meters');
const gradcamBlock = document.getElementById('gradcamBlock');
const gradcamImg = document.getElementById('gradcamImg');

let selectedFile = null;

imageInput.addEventListener('change', (e) => {
  selectedFile = e.target.files[0];
  if (!selectedFile) return;

  preview.src = URL.createObjectURL(selectedFile);
  dropzoneEmpty.style.display = 'none';
  scanStage.style.display = 'block';
  scanLine.style.display = 'none';
  predictBtn.disabled = false;
});

predictBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  predictBtn.disabled = true;
  predictBtn.textContent = 'Scanning...';
  scanLine.style.display = 'block';

  reportEmpty.style.display = 'none';
  reportBody.style.display = 'none';

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const res = await fetch(API_URL, { method: 'POST', body: formData });
    const data = await res.json();
    renderReport(data);
  } catch (err) {
    reportEmpty.style.display = 'block';
    reportEmpty.textContent = "Couldn't reach the scanner. Check that the backend is running.";
  }

  scanLine.style.display = 'none';
  predictBtn.disabled = false;
  predictBtn.textContent = 'Scan leaf';
});

function renderReport(data) {
  const confPct = Math.round(data.confidence * 100);

  verdictClass.textContent = data.predicted_class.replace(/_/g, ' ');
  verdictConf.textContent = confPct + '%';

  warningBanner.style.display = data.low_confidence ? 'flex' : 'none';

  meters.innerHTML = '';
  const sorted = Object.entries(data.all_probabilities).sort((a, b) => b[1] - a[1]);
  const topClass = sorted[0][0];

  sorted.forEach(([cls, prob]) => {
    const pct = Math.round(prob * 100);
    const row = document.createElement('div');
    row.className = 'meter-row';
    row.innerHTML = `
      <div class="meter-top">
        <span class="meter-name">${cls.replace(/_/g, ' ')}</span>
        <span class="meter-val">${pct}%</span>
      </div>
      <div class="meter-track">
        <div class="meter-fill ${cls === topClass ? 'is-top' : ''}" style="width:${pct}%"></div>
      </div>`;
    meters.appendChild(row);
  });

  if (data.gradcam_image) {
    gradcamImg.src = 'data:image/png;base64,' + data.gradcam_image;
    gradcamBlock.style.display = 'block';
  } else {
    gradcamBlock.style.display = 'none';
  }

  reportBody.style.display = 'block';
}
