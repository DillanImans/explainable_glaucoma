import React, { useState, useRef } from 'react';
import MainPage from './MainPage';
import './App.css';

function App() {
  const [step, setStep]         = useState('intro');   // 'intro' | 'loading' | 'main'
  const [analysis, setAnalysis] = useState(null);    
  const [userImage, setUserImage] = useState(null);   
  const fileInputRef            = useRef(null);
  const [overlayUrl, setOverlayUrl] = useState(null);
  const [gptAnalysis, setGptAnalysis] = useState(null);

  const handleFile = async (file) => {
    if (!file) return alert('Please select an image.');

    setUserImage(URL.createObjectURL(file));
    setStep('loading');

    const form = new FormData();
    form.append('image1', file, file.name);

    try {
      const resp = await fetch('http://localhost:5000/infer', {
        method: 'POST',
        body: form,
      });
      if (!resp.ok) throw new Error(`Server error ${resp.status}`);
      const { result, overlays, gptAnalysis } = await resp.json();
      setAnalysis(result);
      setOverlayUrl(`http://localhost:5000${overlays.combined}`);
      setGptAnalysis(gptAnalysis);

      setStep('main');
    } catch (err) {
      console.error('Inference failed:', err);
      alert('Inference failed; check console.');
      setStep('intro');
    }
  };

  const onSelectFile = e => handleFile(e.target.files?.[0]);
  const onDrop      = e => { e.preventDefault(); handleFile(e.dataTransfer.files?.[0]); };
  const onDragOver  = e => e.preventDefault();
  const onButtonClick = () => fileInputRef.current.click();

  if (step === 'loading') {
    return (
      <div className="wrapper">
        <h1 className="title">Processing…</h1>
        <div className="spinner" />
      </div>
    );
  }

  if (step === 'main') {
    return (
      <MainPage
        onBack={() => { setAnalysis(null); setUserImage(null); setGptAnalysis(null); setStep('intro'); }}
        userImage={userImage}
        analysis={analysis}
        overlayImage={overlayUrl}
        gptAnalysis={gptAnalysis}
      />
    );
  }

  // intro
  return (
    <div className="wrapper" onDragOver={onDragOver} onDrop={onDrop}>
      <h1 className="title">Auto Glaucoma Screener</h1>
      <p className="subtitle">Select a fundus image to analyze</p>
      <button className="btn-select" onClick={onButtonClick}>
        Select an image
      </button>
      <p className="drop-hint">or drop an image here</p>
      <input
        type="file"
        accept="image/*"
        onChange={onSelectFile}
        ref={fileInputRef}
        style={{ display: 'none' }}
      />
    </div>
  );
}

export default App;
