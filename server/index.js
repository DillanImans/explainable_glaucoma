'use strict';

require('dotenv').config();
const express       = require('express');
const cors          = require('cors');
const { spawn }     = require('child_process');
const path          = require('path');
const fs            = require('fs');
const multer        = require('multer');
const OpenAI        = require('openai');
const featureInfo   = require('./featureInfo');

const app = express();
const port = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);
app.use('/uploads', express.static(uploadDir));
app.use('/results', express.static(path.join(__dirname, 'inference_results')));

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => {
    const suffix = Date.now() + '-' + Math.round(Math.random()*1e9);
    cb(null, `${suffix}-${file.originalname}`);
  }
});
const upload = multer({ storage });

if (!process.env.OPENAI_API_KEY) {
  console.error('ERROR: OPENAI_API_KEY is not set');
  process.exit(1);
}
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

app.post('/infer', upload.single('image1'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image uploaded' });
  }

  const imagePath     = path.resolve(req.file.path);
  const outputDirName = path.parse(req.file.filename).name;
  const outputDir     = path.join(__dirname, 'inference_results', outputDirName);
  fs.mkdirSync(outputDir, { recursive: true });

  const inferFolder = path.join(__dirname, 'allModelAndData', 'infer');
  const scriptPath  = path.join(inferFolder, 'infer.py');

  const py = spawn('python', [
    scriptPath,
    '--image_path', imagePath,
    '--output_dir', outputDir
  ], { cwd: inferFolder });

  let stderr = '';
  py.stderr.on('data', chunk => {
    stderr += chunk.toString();
    console.error('[infer.py stderr]', chunk.toString());
  });

  py.on('close', async (code) => {
    if (code !== 0) {
      console.error(`infer.py exited with ${code}`, stderr);
      return res.status(500).json({ error: 'Inference failed', details: stderr.trim() });
    }

    const resultPath = path.join(outputDir, 'result.json');
    let result;
    try {
      result = JSON.parse(fs.readFileSync(resultPath, 'utf8'));
    } catch (e) {
      console.error('Failed to read result.json:', e);
      return res.status(500).json({ error: 'Could not read inference output' });
    }

    const overlays = {
      combined: `/results/${outputDirName}/combined.png`,
      sign1:    `/results/${outputDirName}/${result.Top4Signs?.[0]}.png`,
      sign2:    `/results/${outputDirName}/${result.Top4Signs?.[1]}.png`
    };

    const top4 = Object.entries(result)
      .filter(([k]) => !['prediction','prediction_score'].includes(k))
      .sort((a,b) => b[1] - a[1])
      .slice(0,4);

    const bulletLines = top4.map(([code, p]) => {
      const info = featureInfo[code] || { label: code, sig: '' };
      return `• ${info.label} (${(p*100).toFixed(1)}%): ${info.sig}`;
    }).join('\n');

    const promptText = `
Fundus analysis
Prediction: ${result.prediction.toUpperCase()} (${(result.prediction_score*100).toFixed(1)}%)
Top features:
${bulletLines}

Write a single detailed diagnostic paragraph (text only, no lists) 
explaining the pathophysiology and clinical implications based on these findings. 
As a note, the main prediction score is the threshold for 
glaucoma or normal (where 65% or above is glaucoma), 
whilst the feature % are more as a prominence percentage.
    `.trim();

    // call GPT-4o
    let gptAnalysis = '';
    try {
      const gpt = await openai.chat.completions.create({
        model: 'gpt-4o',
        temperature: 0.6,
        max_tokens: 400,
        messages: [{ role: 'user', content: promptText }]
      });
      gptAnalysis = gpt.choices[0].message.content.trim();
    } catch (e) {
      console.error('GPT call failed:', e);
      gptAnalysis = 'Automatic diagnostic paragraph unavailable.';
    }

    // respond
    res.json({
      result,
      overlays,
      gptAnalysis,
      reportUrl: `/results/${outputDirName}/combined.png`
    });
  });
});

app.listen(port, () => {
  console.log(`Server listening on http://localhost:${port}`);
});
