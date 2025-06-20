// src/DocumentDownloader.js
import {
  Document,
  Packer,
  Paragraph,
  TextRun,
} from 'docx';
import { saveAs } from 'file-saver';
import { FEATURE_INFO } from './FeatureInfo';

export default async function downloadReport({
  userImage,
  overlayImage,
  analysis,
  gptParagraph
}) {
  const isGlaucoma = analysis.prediction === 'glaucoma';

  console.log(gptParagraph)
  const top4 = Object.entries(analysis)
    .filter(([k]) => !['prediction','prediction_score'].includes(k))
    .sort((a,b) => b[1] - a[1])
    .slice(0,4)
    .map(([code, prob]) => ({ code, prob }));

  const featureDetails = top4.flatMap(({ code, prob }) => {
    const info = FEATURE_INFO[code] || {};
    return [
      new Paragraph({
        children:[
          new TextRun({ text: info.label || code, bold:true }),
          new TextRun({ text:` (${(prob*100).toFixed(1)}% Prominence)` })
        ],
        spacing:{ after:100 }
      }),
      new Paragraph({ text:`Definition: ${info.definition}` }),
      new Paragraph({ text:`Significance: ${info.significance}`, spacing:{ after:200 } })
    ];
  });

  const doc = new Document({
    sections: [{
      children: [
        new Paragraph({
          children:[ new TextRun({ text:'Auto Glaucoma Screener Report', bold:true, size:32 }) ],
          spacing:{ after:300 }
        }),

        new Paragraph({
          children: [
            new TextRun({ text: 'Prediction: ', bold: true }),
            new TextRun({ text: analysis.prediction.toUpperCase(), bold: true })
          ],
          spacing: { after: 100 }
        }),
        new Paragraph({
          children: [
            new TextRun({ 
              text: `Prediction score: (${(analysis.prediction_score * 100).toFixed(1)}%); `, 
              italics: true 
            }),
            new TextRun({ 
              text: 'Glaucoma if ≥ 65%; normal if < 65%.', 
              italics: true 
            })
          ],
          spacing: { after: 200 }
        }),

        new Paragraph({
          children: [new TextRun({ text: 'Diagnostic Analysis:', bold: true, size: 24 })],
          spacing: { after: 100 }
        }),
        new Paragraph({ text: gptParagraph, spacing: { after: 200 } }),

        new Paragraph({
          children: [new TextRun({ text: 'Top Feature Details:', bold: true, size: 24 })],
          spacing: { after: 100 }
        }),
        ...featureDetails,

        new Paragraph({
          children: [new TextRun({ text: 'Clinical Recommendation:', bold: true, size: 24 })],
          spacing: { before: 300, after: 100 }
        }),
        new Paragraph({
          text: isGlaucoma
            ? 'Consult a glaucoma specialist promptly. Early intervention can significantly slow disease progression.'
            : 'Maintain routine comprehensive eye examinations as scheduled for ongoing monitoring.'
        })
      ]
    }]
  });

  const blob = await Packer.toBlob(doc);
  saveAs(blob, 'glaucoma_report.docx');
}