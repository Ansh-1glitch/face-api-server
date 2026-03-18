const fs = require('fs');
const path = require('path');
const {
  loadModels,
  detectFace,
  detectScreenSpoof,
  extractFaceEmbedding,
  compareEmbeddings
} = require('./faceEngine');

async function testModel() {
  console.log('🤖 Starting Local AI Model Tester...\n');

  // 1. Load the AI Models
  await loadModels();

  // 2. Read the test images from the folder
  const testImagePath1 = path.join(__dirname, 'test1.jpg');
  const testImagePath2 = path.join(__dirname, 'test2.jpg');

  if (!fs.existsSync(testImagePath1)) {
    console.error(`\n❌ Please place a photo named "test1.jpg" inside the server folder: ${__dirname}`);
    process.exit(1);
  }

  const base64Image1 = fs.readFileSync(testImagePath1, 'base64');
  let base64Image2 = null;

  if (fs.existsSync(testImagePath2)) {
    base64Image2 = fs.readFileSync(testImagePath2, 'base64');
  }

  try {
    console.log('\n--- 📸 Testing Image 1 (test1.jpg) ---');
    
    // Test Face Detection
    console.log('⏳ Detecting Face...');
    const hasFace1 = await detectFace(base64Image1);
    console.log(hasFace1 ? '✅ Face Found' : '❌ No Face Found');

    if (!hasFace1) return;

    // Test Anti-Spoofing
    console.log('\n⏳ Checking for Spoof (Screen/Photo)...');
    const spoof1 = await detectScreenSpoof(base64Image1);
    console.log(spoof1.isReal ? '✅ PASS: Real Face' : `❌ FAIL: Fake Face (${spoof1.reason})`);
    console.log(`   (Saturation: ${spoof1.saturation}, Variance: ${spoof1.variance})`);

    // Test Embedding Extraction
    console.log('\n⏳ Extracting Face Data (LBP+Gradient)...');
    const embedding1 = await extractFaceEmbedding(base64Image1);
    console.log(`✅ Success: Generated ${embedding1.length} data points`);

    // Compare with Image 2 (Optional)
    if (base64Image2) {
      console.log('\n--- 📸 Testing Comparison with (test2.jpg) ---');
      const hasFace2 = await detectFace(base64Image2);
      if (hasFace2) {
        const embedding2 = await extractFaceEmbedding(base64Image2);
        const score = compareEmbeddings(embedding1, embedding2);
        const isMatch = score >= 0.75;
        console.log(`\n⚖️ Match Score: ${score.toFixed(4)} (Threshold: 0.75)`);
        console.log(isMatch ? '✅ SUCCESS: Faces Match!' : '❌ FAIL: Different Faces (Or poor quality image)');
      }
    } else {
      console.log('\n💡 Tip: Add a "test2.jpg" image to this folder to test if two different images match!');
    }

  } catch (error) {
    console.error('❌ Test failed with error:', error);
  }
}

testModel();
