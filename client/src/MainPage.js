import React, { useState, useRef, useEffect } from 'react';
import './MainPage.css';
import PoiToggleSprite from './PoiToggleSprite';
import downloadReport from './DocumentDownloader';
import { FEATURE_INFO } from './FeatureInfo';

export default function MainPage({ onBack, userImage, analysis, overlayImage, gptAnalysis }) {
  const [overlayHover, setOverlayHover] = useState(false);
  const [overlayActive, setOverlayActive] = useState(false);
  const [atBottom, setAtBottom] = useState(false);
  const mainRef = useRef(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    const container = mainRef.current;
    if (!container) return;

    const obs = new IntersectionObserver(
      ([entry]) => setAtBottom(entry.intersectionRatio >= 0.9),
      { root: container, threshold: [0, 0.9] }
    );

    if (bottomRef.current) obs.observe(bottomRef.current);
    return () => obs.disconnect();
  }, []);

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

    const handleDownload = async () => {
    try {
      const [resUser, resOverlay] = await Promise.all([
        fetch(userImage),
        fetch(overlayImage),
      ]);

      if (!resUser.ok || !resOverlay.ok) {
        throw new Error('Failed to fetch one of the images');
      }

      const [blobUser, blobOverlay] = await Promise.all([
        resUser.blob(),
        resOverlay.blob(),
      ]);

      const triggerSave = (blob, filename) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      };

      triggerSave(blobUser, 'fundus.png');
      triggerSave(blobOverlay, 'overlay.png');

      await downloadReport({
        userImage,
        overlayImage,
        analysis,
        gptParagraph: gptAnalysis,
      });
    } catch (e) {
      console.error('Download failed:', e);
    }
  };

    const isGlaucoma = analysis.prediction === 'glaucoma';
    const prelimText = isGlaucoma
    ? 'Our system has analyzed your retinal fundus image and determined that there are strong indicators consistent with glaucoma.'
    : 'Our system has analyzed your retinal fundus image and did not detect any findings suggestive of glaucomatous optic neuropathy at this time.';

    const recommendationText = isGlaucoma
    ? 'We recommend consulting a glaucoma specialist promptly. Early intervention can significantly slow disease progression and preserve vision.'
    : 'Maintain your routine comprehensive eye examinations as scheduled, ensuring ongoing monitoring for any future changes in optic-nerve or retinal health.';
    const featureEntries = Object.entries(analysis)
    .filter(([k]) => !(k === 'prediction' || k === 'prediction_score'))
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4);


  return (
    <div ref={mainRef} className="main-page">
      {/* Top section */}
      <div className="top-section">
        <button className="back-btn" onClick={onBack}>
          <img src="icons/arrow.png" className="back-icon" alt="Back" />
        </button>

        <div className="content">
          <div className="left-panel">
            <div className="image-container">
              <img src={userImage} alt="Fundus" />
              <img
                src={overlayImage}
                className={`image-overlay${overlayActive ? ' active' : overlayHover ? ' hover' : ''}`}
                alt="Overlay"
                onMouseEnter={() => setOverlayHover(true)}
                onMouseLeave={() => setOverlayHover(false)}
                onClick={() => setOverlayActive(a => !a)}
              />
            </div>

            <div className="prediction">
              <span className="label">Prediction:</span>
              <span className={`result ${analysis.prediction === 'glaucoma' ? 'detected' : 'normal'}`}>
                {analysis.prediction === 'glaucoma' ? 'Detected' : 'Normal'}
              </span>
            </div>
            <div className="confidence-text">
              Prediction score: {(analysis.prediction_score * 100).toFixed(1)}%
              <p className="smallertext">
                Glaucoma if ≥ 65%; normal if &lt; 65%.
              </p>
              </div>
          </div>

          <div className="right-panel">
            <div className="signs-group">
              <h2 className="signs-title">Signs</h2>
              <div className="sign-list">
                {featureEntries.map(([feature, conf]) => (
                  <div className="sign-item" key={feature}>
                    <div className="feature">{feature}</div>
                    <div className="conf">{(conf * 100).toFixed(1)}%</div>
                  </div>
                ))}
              </div>
              <p className="signs_caption">Higher % = more likely the sign is present</p>
            </div>

            <div className="poi-toggle">
              <div
                className="poi-sprite"
                onMouseEnter={() => setOverlayHover(true)}
                onMouseLeave={() => setOverlayHover(false)}
                onClick={() => setOverlayActive(a => !a)}
              >
                <PoiToggleSprite src="/icons/poi.png" />
              </div>
              <span className="poi-label">Toggle Point-Of-Interest</span>
            </div>
          </div>
        </div>

        <button
          className="scroll-down-btn"
          onClick={scrollToBottom}
          aria-label="Scroll Down"
          style={{
            opacity: atBottom ? 0 : 1,
            pointerEvents: atBottom ? 'none' : 'auto',
            transition: 'opacity 0.2s ease',
          }}
        >
          <img src="/icons/arrow-down.png" alt="↓" />
        </button>
      </div>

      {/* Bottom section */}
      <div ref={bottomRef} className="bottom-section">

      <div className="gpt-box">
        <h2>Preliminary Assessment</h2>
        <p>{prelimText}</p>

        <br />

        <h3>Detected Features</h3>
        <ul>
            {featureEntries.map(([code, prob]) => {
            const info = FEATURE_INFO[code] || {};
            return (
                <li key={code}>
                <strong>{info.label || code}</strong> ({(prob * 100).toFixed(1)}% Prominence)
                <br />
                <em>Clinical Definition:</em> {info.definition}
                <br />
                <em>Clinical Significance:</em> {info.significance}
                </li>
            );
            })}
        </ul>

        <br />

        <h3>Diagnostic Analysis</h3>
         <p>{gptAnalysis}</p>

        <br />

        <h3>Clinical Recommendation</h3>
        <p>{recommendationText}</p>
        </div>  

        <div
          className="icon-area"
          onClick={handleDownload}
        >
          <img src="icons/document.png" alt="Docs" />
          <p className="icon-areap">Download Report</p>
        </div>
      </div>
    </div>
  );
}
