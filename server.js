const express = require('express');
const cors = require('cors');
const multer = require('multer');
const admin = require('firebase-admin');
const {
  loadModels, isModelsLoaded, extractFaceEmbedding,
  detectFace, detectScreenSpoof, compareEmbeddings, averageDescriptors
} = require('./faceEngine');

// ─── Firebase Setup ─────────────────────────────────────────────────────────

// Initialize Firebase Admin (Requires service account key in production)
// For local dev, you can set GOOGLE_APPLICATION_CREDENTIALS in .env
// Example: process.env.GOOGLE_APPLICATION_CREDENTIALS = './serviceAccountKey.json';
try {
  admin.initializeApp();
  console.log('✅ Firebase Admin initialized');
} catch (e) {
  console.warn('⚠️ Firebase Admin not initialized. Firestore saving will fail if not configured.');
}
const db = admin.firestore();

// ─── Express App Setup ──────────────────────────────────────────────────────

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Multer for multipart/form-data (in case clients send files directly)
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });

const THRESHOLD = 0.75; // Same as mobile app

// ─── API Routes ─────────────────────────────────────────────────────────────

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', modelsLoaded: isModelsLoaded() });
});

/**
 * Helper to get base64 from either JSON body or multipart file
 */
function getBase64Image(req) {
  if (req.file) {
    return req.file.buffer.toString('base64');
  } else if (req.body && req.body.image) {
    // Remove data URI prefix if present
    return req.body.image.replace(/^data:image\/\w+;base64,/, '');
  }
  return null;
}

/**
 * /api/register
 * Expects: studentUID, image1, image2, image3 (base64 strings)
 * Saves averaged descriptor to Firestore.
 */
app.post('/api/register', async (req, res) => {
  try {
    const { studentUID, name, rollNo, email, image1, image2, image3 } = req.body;
    
    if (!studentUID || !name || !rollNo || !email || !image1 || !image2 || !image3) {
      return res.status(400).json({ error: 'Missing required parameters (studentUID, name, rollNo, email, image1, image2, image3)' });
    }

    if (!isModelsLoaded()) await loadModels();

    const embeddings = [];
    for (const img of [image1, image2, image3]) {
      const faceFound = await detectFace(img);
      if (!faceFound) return res.status(400).json({ error: 'No face detected in one or more images' });
      
      const emb = await extractFaceEmbedding(img);
      if (!emb) return res.status(500).json({ error: 'Failed to extract embedding' });
      embeddings.push(emb);
    }

    const averaged = averageDescriptors(embeddings);
    const descriptorArray = Array.from(averaged);

    // Save to Firestore
    await db.collection('users').doc(studentUID).set({
      name,
      rollNo,
      email,
      faceDescriptor: descriptorArray,
      registeredAt: admin.firestore.FieldValue.serverTimestamp(),
      testMode: false,
      embeddingVersion: 'lbp_grad_v1',
    }, { merge: true });

    res.json({ success: true, message: 'Face registered successfully', descriptorLength: descriptorArray.length });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * /api/verify
 * Expects: studentUID, image (base64 or multipart file)
 * Returns verification result.
 */
app.post('/api/verify', upload.single('image'), async (req, res) => {
  try {
    const studentUID = req.body.studentUID;
    const base64Image = getBase64Image(req);

    if (!studentUID || !base64Image) {
      return res.status(400).json({ error: 'Missing studentUID or image' });
    }

    if (!isModelsLoaded()) await loadModels();

    // 1. Check face presence
    const faceFound = await detectFace(base64Image);
    if (!faceFound) {
      return res.status(400).json({ match: false, error: 'No face detected' });
    }

    // 2. Spoof check
    const spoofResult = await detectScreenSpoof(base64Image);
    if (!spoofResult.isReal) {
      return res.status(403).json({ match: false, error: 'Spoof detected', reason: spoofResult.reason, spoof: true });
    }

    // 3. Get live embedding
    const liveEmbedding = await extractFaceEmbedding(base64Image);
    if (!liveEmbedding) {
      return res.status(500).json({ match: false, error: 'Failed to extract face embedding' });
    }

    // 4. Fetch stored descriptor from Firestore
    const userDoc = await db.collection('users').doc(studentUID).get();
    if (!userDoc.exists || !userDoc.data().faceDescriptor) {
      return res.status(404).json({ match: false, error: 'Student face not registered' });
    }

    const storedDescriptor = new Float32Array(userDoc.data().faceDescriptor);

    // 5. Compare
    const score = compareEmbeddings(storedDescriptor, liveEmbedding);
    const match = score >= THRESHOLD;

    res.json({
      match,
      score: parseFloat(score.toFixed(4)),
      threshold: THRESHOLD,
      success: match
    });
  } catch (error) {
    console.error('Verification error:', error);
    res.status(500).json({ error: error.message });
  }
});

// ─── Start Server ───────────────────────────────────────────────────────────

const PORT = process.env.PORT || 3000;

// Load models before starting
loadModels().then(() => {
  app.listen(PORT, () => {
    console.log(`🚀 Face API Server running on port ${PORT}`);
  });
}).catch(err => {
  console.error('❌ Failed to load models on startup:', err);
});
