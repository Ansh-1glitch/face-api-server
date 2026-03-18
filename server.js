const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const admin = require('firebase-admin');
const {
  loadModels, isModelsLoaded, extractFaceEmbedding,
  detectFace, detectScreenSpoof, compareEmbeddings, averageDescriptors
} = require('./faceEngine');

// ─── Configuration ──────────────────────────────────────────────────────────

// SAAMS Backend URL (the actual database/student backend)
const SAAMS_BASE_URL = process.env.SAAMS_BASE_URL || 'https://your-saams-backend.onrender.com';

try {
  if (process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON) {
    const serviceAccount = JSON.parse(process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON);
    admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
  } else {
    admin.initializeApp();
  }
  console.log('✅ Firebase Admin initialized');
} catch (e) {
  console.warn('⚠️ Firebase Admin not initialized.');
}
const db = admin.firestore();

// ─── Express App Setup ──────────────────────────────────────────────────────

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });
const THRESHOLD = 0.75; // Same as mobile app

// Helper
function getBase64Image(req) {
  if (req.file) return req.file.buffer.toString('base64');
  if (req.body && req.body.image) return req.body.image.replace(/^data:image\/\w+;base64,/, '');
  return null;
}

// ─── API Routes ─────────────────────────────────────────────────────────────

app.get('/health', (req, res) => {
  res.json({ status: 'ok', modelsLoaded: isModelsLoaded() });
});

/**
 * /api/register
 * 1. Takes Firebase ID Token + 3 images.
 * 2. Fetches student details directly from SAAMS db.
 * 3. Saves face embedding + student details into Firebase.
 */
app.post('/api/register', async (req, res) => {
  try {
    const { token, image1, image2, image3 } = req.body;
    
    if (!token || !image1 || !image2 || !image3) {
      return res.status(400).json({ error: 'Missing required parameters (token, image1, image2, image3)' });
    }

    // 1. Fetch student details from SAAMS using the token
    let studentData;
    try {
      const profileRes = await axios.get(`${SAAMS_BASE_URL}/api/auth/profile`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      studentData = profileRes.data.data;
    } catch (e) {
      return res.status(401).json({ error: 'Failed to fetch student details. Invalid token or SAAMS offline.', details: e.message });
    }

    const { uid: studentUID, name, rollNumber, email } = studentData;

    // 2. Process faces
    if (!isModelsLoaded()) await loadModels();
    const embeddings = [];
    for (const img of [image1, image2, image3]) {
      const faceFound = await detectFace(img);
      if (!faceFound) return res.status(400).json({ error: 'No face detected in one or more images' });
      
      const emb = await extractFaceEmbedding(img);
      if (!emb) return res.status(500).json({ error: 'Failed to extract face embedding' });
      embeddings.push(emb);
    }

    const averaged = averageDescriptors(embeddings);
    const descriptorArray = Array.from(averaged);

    // 3. Save to Firebase
    await db.collection('users').doc(studentUID).set({
      name,
      rollNo: rollNumber,  // SAAMS calls it rollNumber
      email,
      faceDescriptor: descriptorArray,
      registeredAt: admin.firestore.FieldValue.serverTimestamp(),
      testMode: false,
      embeddingVersion: 'lbp_grad_v1',
    }, { merge: true });

    res.json({ success: true, message: 'Face registered successfully', studentId: studentUID });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * /api/attendance/mark (Middleman endpoint)
 * 1. Verifies the face (Spoofing check + Match check).
 * 2. If valid, forwards the exact request to SAAMS /api/attendance/mark.
 * Requires: token, image, along with all attendance payload (sessionId, method, etc.)
 */
app.post('/api/attendance/mark', upload.single('image'), async (req, res) => {
  try {
    const { token, ...attendancePayload } = req.body;
    const base64Image = getBase64Image(req);

    if (!token || !base64Image) {
      return res.status(400).json({ error: 'Missing token or image' });
    }

    // 1. Get student UID by verifying the token with SAAMS locally, 
    // or we could just fetch the profile again to be safe.
    let studentData;
    try {
      const profileRes = await axios.get(`${SAAMS_BASE_URL}/api/auth/profile`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      studentData = profileRes.data.data;
    } catch (e) {
      return res.status(401).json({ error: 'Invalid token' });
    }

    const studentUID = studentData.uid;

    if (!isModelsLoaded()) await loadModels();

    // 2. Verify Face Presence
    const faceFound = await detectFace(base64Image);
    if (!faceFound) {
      return res.status(400).json({ match: false, error: 'No face detected' });
    }

    // 3. Spoof Check
    const spoofResult = await detectScreenSpoof(base64Image);
    if (!spoofResult.isReal) {
      return res.status(403).json({ match: false, error: 'Spoof detected', reason: spoofResult.reason, spoof: true });
    }

    // 4. Live Embedding extraction
    const liveEmbedding = await extractFaceEmbedding(base64Image);
    
    // 5. Check against DB
    const userDoc = await db.collection('users').doc(studentUID).get();
    if (!userDoc.exists || !userDoc.data().faceDescriptor) {
      return res.status(404).json({ match: false, error: 'Student face not registered' });
    }

    const storedDescriptor = new Float32Array(userDoc.data().faceDescriptor);
    const score = compareEmbeddings(storedDescriptor, liveEmbedding);
    const match = score >= THRESHOLD;

    if (!match) {
      return res.status(403).json({ match: false, error: 'Face does not match registered face', score });
    }

    // 6. FACE VERIFIED! Forward the request to SAAMS attendance endpoint.
    try {
      const saamsResponse = await axios.post(`${SAAMS_BASE_URL}/api/attendance/mark`, attendancePayload, {
        headers: { Authorization: `Bearer ${token}` }
      });

      // Forward SAAMS response directly to client + add faceMatch flag
      return res.status(saamsResponse.status).json({
        ...saamsResponse.data,
        faceMatch: true,
        score
      });

    } catch (saamsError) {
      // Forward SAAMS error correctly
      const statusCode = saamsError.response ? saamsError.response.status : 500;
      const errorData = saamsError.response ? saamsError.response.data : { error: 'SAAMS backend unreachable' };
      return res.status(statusCode).json(errorData);
    }

  } catch (error) {
    console.error('Attendance proxy error:', error);
    res.status(500).json({ error: error.message });
  }
});

// ─── Start Server ───────────────────────────────────────────────────────────

const PORT = process.env.PORT || 3000;

loadModels().then(() => {
  app.listen(PORT, () => console.log(`🚀 AI Middleman Server running on port ${PORT}`));
}).catch(err => console.error('❌ Failed to load models:', err));
