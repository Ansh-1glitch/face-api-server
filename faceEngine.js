/**
 * faceEngine.js — Face embedding + spoofing logic for Node.js
 * Uses @tensorflow/tfjs-node + sharp for image processing
 * Same LBP+gradient algorithm as the mobile app
 */

const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');
const path = require('path');

let blazefaceModel = null;
let isReady = false;

const MODEL_PATH = `file://${path.join(__dirname, 'models', 'blazeface-model.json')}`;

async function loadModels() {
  if (isReady) return;
  blazefaceModel = await tf.loadGraphModel(MODEL_PATH);
  isReady = true;
  console.log('✅ BlazeFace model loaded');
}

function isModelsLoaded() { return isReady; }

// ─── Image helpers ────────────────────────────────────────────────────────────

/** Convert base64 string to sharp image buffer */
function base64ToBuffer(base64) {
  const base64Data = base64.replace(/^data:image\/\w+;base64,/, '');
  return Buffer.from(base64Data, 'base64');
}

/** Resize image buffer and return raw RGB pixel data */
async function bufferToPixels(imageBuffer, size) {
  const { data, info } = await sharp(imageBuffer)
    .resize(size, size, { fit: 'fill' })
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { pixels: data, info };
}

/** Get grayscale Uint8Array from raw RGB buffer */
function rgbToGray(rgbData, width, height) {
  const gray = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const r = rgbData[i * 3];
    const g = rgbData[i * 3 + 1];
    const b = rgbData[i * 3 + 2];
    gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  }
  return gray;
}

// ─── LBP Descriptor ──────────────────────────────────────────────────────────

function lbpCode(pixels, w, x, y) {
  const c = pixels[y * w + x];
  const ns = [
    pixels[(y-1)*w+(x-1)], pixels[(y-1)*w+x], pixels[(y-1)*w+(x+1)],
    pixels[y*w+(x+1)],
    pixels[(y+1)*w+(x+1)], pixels[(y+1)*w+x], pixels[(y+1)*w+(x-1)],
    pixels[y*w+(x-1)],
  ];
  let code = 0;
  for (let i = 0; i < 8; i++) if ((ns[i] || 0) >= c) code |= (1 << i);
  return code;
}

function cellHistogram(pixels, w, x0, y0, cw, ch, bins) {
  const hist = new Float32Array(bins).fill(0);
  let count = 0;
  for (let y = y0 + 1; y < y0 + ch - 1; y++) {
    for (let x = x0 + 1; x < x0 + cw - 1; x++) {
      if (x > 0 && x < w - 1 && y > 0 && y < w - 1) {
        hist[Math.floor(lbpCode(pixels, w, x, y) / 256 * bins)]++;
        count++;
      }
    }
  }
  if (count > 0) for (let i = 0; i < bins; i++) hist[i] /= count;
  return hist;
}

function computeLBP(pixels, imgSize, gx = 7, gy = 7, bins = 16) {
  const cw = Math.floor(imgSize / gx);
  const ch = Math.floor(imgSize / gy);
  const out = new Float32Array(gx * gy * bins);
  let idx = 0;
  for (let row = 0; row < gy; row++) {
    for (let col = 0; col < gx; col++) {
      const h = cellHistogram(pixels, imgSize, col * cw, row * ch, cw, ch, bins);
      for (let b = 0; b < bins; b++) out[idx++] = h[b];
    }
  }
  return out;
}

// ─── Gradient Descriptor ─────────────────────────────────────────────────────

function computeGradients(pixels, imgSize, gx = 7, gy = 7) {
  const cw = Math.floor(imgSize / gx);
  const ch = Math.floor(imgSize / gy);
  const out = new Float32Array(gx * gy * 4);
  let idx = 0;
  for (let row = 0; row < gy; row++) {
    for (let col = 0; col < gx; col++) {
      let sumH = 0, sumV = 0, sumM = 0, count = 0;
      for (let y = row * ch + 1; y < (row + 1) * ch - 1; y++) {
        for (let x = col * cw + 1; x < (col + 1) * cw - 1; x++) {
          if (x > 0 && x < imgSize - 1 && y > 0 && y < imgSize - 1) {
            const dx = (pixels[y * imgSize + x + 1] || 0) - (pixels[y * imgSize + x - 1] || 0);
            const dy = (pixels[(y + 1) * imgSize + x] || 0) - (pixels[(y - 1) * imgSize + x] || 0);
            sumH += Math.abs(dx); sumV += Math.abs(dy);
            sumM += Math.sqrt(dx * dx + dy * dy); count++;
          }
        }
      }
      out[idx++] = count > 0 ? sumH / count / 255 : 0;
      out[idx++] = count > 0 ? sumV / count / 255 : 0;
      out[idx++] = count > 0 ? sumM / count / 255 : 0;
      out[idx++] = count > 0 ? sumH / (sumV + 1e-6) : 1;
    }
  }
  return out;
}

function l2Normalize(arr) {
  let norm = 0;
  for (let i = 0; i < arr.length; i++) norm += arr[i] * arr[i];
  norm = Math.sqrt(norm);
  if (norm > 1e-6) for (let i = 0; i < arr.length; i++) arr[i] /= norm;
  return arr;
}

// ─── Main Public Functions ────────────────────────────────────────────────────

/**
 * Extract a 980-value LBP+gradient face embedding from a base64 image string.
 * @param {string} base64Image
 * @returns {Float32Array|null}
 */
async function extractFaceEmbedding(base64Image) {
  try {
    const buf = base64ToBuffer(base64Image);

    // Resize to 320x320, crop center 80%, resize to 64x64
    const resized = await sharp(buf).resize(320, 320, { fit: 'fill' }).toBuffer();
    const margin = Math.round(320 * 0.1);
    const cropSize = 320 - margin * 2;
    const cropped = await sharp(resized)
      .extract({ left: margin, top: margin, width: cropSize, height: cropSize })
      .resize(64, 64, { fit: 'fill' })
      .raw()
      .toBuffer();

    const pixels = rgbToGray(cropped, 64, 64);
    const lbp = computeLBP(pixels, 64, 7, 7, 16);
    const grad = computeGradients(pixels, 64, 7, 7);
    const combined = new Float32Array(lbp.length + grad.length);
    combined.set(lbp, 0);
    combined.set(grad, lbp.length);
    l2Normalize(combined);
    return combined;
  } catch (err) {
    console.error('extractFaceEmbedding error:', err.message);
    return null;
  }
}

/**
 * Detect if a face is present in the image using BlazeFace.
 * @param {string} base64Image
 * @returns {boolean}
 */
async function detectFace(base64Image) {
  if (!blazefaceModel) throw new Error('Model not loaded');
  let tensor = null;
  let preprocessed = null;
  try {
    const buf = base64ToBuffer(base64Image);
    const { pixels: rgbPixels } = await bufferToPixels(buf, 128);
    // Build [1, 128, 128, 3] tensor normalized to [0, 1]
    const floatData = new Float32Array(128 * 128 * 3);
    for (let i = 0; i < 128 * 128 * 3; i++) floatData[i] = rgbPixels[i] / 255.0;
    tensor = tf.tensor4d(floatData, [1, 128, 128, 3]);

    const output = await blazefaceModel.executeAsync(tensor);
    let scoresData;
    if (Array.isArray(output)) {
      scoresData = await output[0].data();
      output.forEach(t => t && t.dispose());
    } else {
      scoresData = await output.data();
      output.dispose();
    }
    for (let i = 0; i < scoresData.length; i++) {
      if (scoresData[i] > 0.5) return true;
    }
    return false;
  } catch (err) {
    console.error('detectFace error:', err.message);
    return false;
  } finally {
    if (tensor) tensor.dispose();
  }
}

/**
 * Basic screen/photo spoof check.
 * @param {string} base64Image
 * @returns {{ isReal: boolean, reason: string }}
 */
async function detectScreenSpoof(base64Image) {
  try {
    const buf = base64ToBuffer(base64Image);
    const { pixels: rgbPixels } = await bufferToPixels(buf, 128);
    const total = 128 * 128;

    let satSum = 0, brightnessValues = [];
    for (let i = 0; i < total; i++) {
      const r = rgbPixels[i * 3] / 255;
      const g = rgbPixels[i * 3 + 1] / 255;
      const b = rgbPixels[i * 3 + 2] / 255;
      const maxC = Math.max(r, g, b);
      const minC = Math.min(r, g, b);
      satSum += maxC > 0.01 ? (maxC - minC) / maxC : 0;
      brightnessValues.push(0.299 * r + 0.587 * g + 0.114 * b);
    }

    const meanSat = satSum / total;
    const meanB = brightnessValues.reduce((a, b) => a + b, 0) / total;
    const variance = brightnessValues.reduce((a, b) => a + (b - meanB) ** 2, 0) / total;

    const isHighSat = meanSat > 0.42;
    const isLowVar = variance < 0.004;
    const isReal = !isHighSat && !isLowVar;

    return {
      isReal,
      saturation: +meanSat.toFixed(3),
      variance: +variance.toFixed(5),
      reason: isHighSat ? 'High saturation (screen/photo)' : isLowVar ? 'Too uniform (flat surface)' : 'Appears real',
    };
  } catch (err) {
    return { isReal: true, reason: 'Check skipped' };
  }
}

/**
 * Compare two descriptors — cosine similarity 0–1.
 */
function compareEmbeddings(e1, e2) {
  if (!e1 || !e2 || e1.length !== e2.length) return 0;
  let dot = 0, n1 = 0, n2 = 0;
  for (let i = 0; i < e1.length; i++) {
    dot += e1[i] * e2[i]; n1 += e1[i] * e1[i]; n2 += e2[i] * e2[i];
  }
  return Math.max(0, Math.min(1, dot / (Math.sqrt(n1) * Math.sqrt(n2) + 1e-6)));
}

/**
 * Average an array of Float32Array descriptors into one.
 */
function averageDescriptors(descriptors) {
  const len = descriptors[0].length;
  const sum = new Float32Array(len);
  for (const d of descriptors) for (let i = 0; i < len; i++) sum[i] += d[i];
  for (let i = 0; i < len; i++) sum[i] /= descriptors.length;
  return sum;
}

module.exports = {
  loadModels, isModelsLoaded,
  extractFaceEmbedding, detectFace, detectScreenSpoof,
  compareEmbeddings, averageDescriptors,
};
