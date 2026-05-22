import { useEffect, useRef, useState } from 'react';
import { FaceLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';
import { GoogleGenAI, Type } from '@google/genai';
import { Activity, AudioLines, Camera, RefreshCw, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const DEADZONE = 2.0;
const LERP_VAL = 0.05;

export default function VibeVizTab() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [faceLandmarker, setFaceLandmarker] = useState<FaceLandmarker | null>(null);
  const [webcamRunning, setWebcamRunning] = useState(false);
  const webcamRunningRef = useRef(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);

  const [emotions, setEmotions] = useState({
    Smile: 0,
    Laugh: 0,
    Sad: 0,
    Angry: 0,
    Surprised: 0,
    Fear: 0,
  });
  const [dominantEmotion, setDominantEmotion] = useState('Neutral');
  const [vibeValue, setVibeValue] = useState(0.5);
  const vibeValueRef = useRef<number>(0.5);

  const [timeWindow, setTimeWindow] = useState<number>(15);
  const timeWindowRef = useRef<number>(15);
  
  // Update ref when state changes
  useEffect(() => {
    timeWindowRef.current = timeWindow;
  }, [timeWindow]);

  const [vibeHistory, setVibeHistory] = useState<{time: number, score: number | null}[]>([]);

  interface AnalysisResult {
    summary: string;
    details: string;
  }

  const [geminiAnalysis, setGeminiAnalysis] = useState<AnalysisResult | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const [isRecording, setIsRecording] = useState(false);
  const isRecordingRef = useRef(false);
  const [showLandmarks, setShowLandmarks] = useState(false);
  const showLandmarksRef = useRef(false);
  const [recordFrequency, setRecordFrequency] = useState(100);
  const recordFrequencyRef = useRef(100);
  const lastRecordTimeRef = useRef<number>(0);
  const [numFacesDetected, setNumFacesDetected] = useState(0);
  const numFacesDetectedRef = useRef(0);

  // Refs for animation loop and state that doesn't need re-renders
  const requestRef = useRef<number>(0);
  const lastVideoTimeRef = useRef<number>(-1);
  const baseValuesRef = useRef<Record<number, { biu: number, bd: number, bid: number, ns: number }>>({});
  const latestBlendshapesRef = useRef<Record<number, any[]>>({});
  const smoothedScoresRef = useRef<Record<number, any>>({});
  const smoothedMoodValueRef = useRef<Record<number, number>>({});
  const smoothedHUDRef = useRef<Record<number, any>>({});
  const nextFaceIdRef = useRef(0);
  const trackedFacesRef = useRef<Array<{ id: number, x: number, y: number, lastSeen: number }>>([]);
  const lastDominantEmotionRef = useRef('Neutral');
  
  // Store recent blendshapes for Gemini analysis (last 50 frames)
  const recentBlendshapesRef = useRef<any[][]>([]);

  useEffect(() => {
    let active = true;
    async function setupModel() {
      try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm'
        );
        const landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: 'GPU',
          },
          outputFaceBlendshapes: true,
          runningMode: 'VIDEO',
          numFaces: 4,
        });
        if (active) {
          setFaceLandmarker(landmarker);
          setIsModelLoading(false);
        } else {
          landmarker.close();
        }
      } catch (error) {
        console.error('Engine Failure', error);
        if (active) setIsModelLoading(false);
      }
    }
    setupModel();

    return () => {
      active = false;
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  // Close landmarker on unmount
  useEffect(() => {
    return () => {
      if (faceLandmarker) faceLandmarker.close();
    };
  }, [faceLandmarker]);

  useEffect(() => {
    if (!webcamRunning) return;
    const interval = setInterval(() => {
      const now = Date.now();
      const score = numFacesDetectedRef.current > 0 ? vibeValueRef.current : null;
      setVibeHistory(prev => {
        const cutoff = now - timeWindow * 1000;
        const filtered = prev.filter(p => p.time >= cutoff);
        return [...filtered, { time: now, score }];
      });
    }, 50); // 20 fps for smoother scrolling
    return () => clearInterval(interval);
  }, [webcamRunning, timeWindow]);

  const startCamera = async () => {
    if (!faceLandmarker) return;
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setWebcamRunning(true);
          webcamRunningRef.current = true;
          predictWebcam();
        };
      }
    } catch (err: any) {
      console.error('Error accessing webcam', err);
      setCameraError(err.message || 'Permission denied');
    }
  };

  useEffect(() => {
    if (faceLandmarker && !webcamRunningRef.current) {
      startCamera();
    }
  }, [faceLandmarker]);

  const stopCamera = () => {
    setWebcamRunning(false);
    webcamRunningRef.current = false;
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.resetTransform();
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
    if (requestRef.current) cancelAnimationFrame(requestRef.current);
  };

  const toggleCamera = () => {
    setGeminiAnalysis(null);
    if (webcamRunning) {
      stopCamera();
    } else {
      startCamera();
    }
  };

  const calibrateBaseline = () => {
    for (const [faceId, blendshapes] of Object.entries(latestBlendshapesRef.current) as [string, any[]][]) {
      const getS = (n: string) => blendshapes.find((b: any) => b.categoryName === n)?.score || 0;
      baseValuesRef.current[Number(faceId)] = {
        biu: getS('browInnerUp'),
        bd: (getS('browDownLeft') + getS('browDownRight')) / 2,
        bid: (getS('browInnerDownLeft') + getS('browInnerDownRight')) / 2,
        ns: (getS('noseSneerLeft') + getS('noseSneerRight')) / 2,
      };
    }
  };

  const predictWebcam = () => {
    if (!videoRef.current || !canvasRef.current || !faceLandmarker) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (video.readyState >= 2 && ctx) {
      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      if (lastVideoTimeRef.current !== video.currentTime) {
        lastVideoTimeRef.current = video.currentTime;
        try {
          const results = faceLandmarker.detectForVideo(video, performance.now());
          ctx.resetTransform();
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          if (results && results.faceLandmarks?.length > 0 && results.faceBlendshapes?.length > 0) {
            const numFaces = results.faceLandmarks.length;
            if (numFacesDetectedRef.current !== numFaces) {
              numFacesDetectedRef.current = numFaces;
              setNumFacesDetected(numFaces);
            }

            // Record the average blendshapes of all detected faces for Gemini analysis
            if (isRecordingRef.current && numFaces > 0) {
              const now = performance.now();
              if (now - lastRecordTimeRef.current >= recordFrequencyRef.current) {
                const avgCategories = results.faceBlendshapes[0].categories.map((cat, idx) => {
                  let sum = 0;
                  for (let i = 0; i < numFaces; i++) {
                    sum += results.faceBlendshapes[i].categories[idx].score;
                  }
                  return { ...cat, score: sum / numFaces };
                });
                recentBlendshapesRef.current.push(avgCategories);
                
                const maxFrames = (timeWindowRef.current * 1000) / recordFrequencyRef.current;
                if (recentBlendshapesRef.current.length > maxFrames) {
                  recentBlendshapesRef.current = recentBlendshapesRef.current.slice(-maxFrames);
                }
                
                lastRecordTimeRef.current = now;
              }
            }

            let totalMoodTarget = 0;
            const totalSmoothed: Record<string, number> = { Smile: 0, Laugh: 0, Sad: 0, Angry: 0, Surprised: 0, Fear: 0, Sleepy: 0 };

            const now = performance.now();
            const currentFaces = [];
            for (let i = 0; i < numFaces; i++) {
              const nose = results.faceLandmarks[i][1]; // Nose tip
              currentFaces.push({
                index: i,
                x: nose.x,
                y: nose.y,
                blendshapes: results.faceBlendshapes[i].categories,
                landmarks: results.faceLandmarks[i]
              });
            }

            const matchedFaces = [];
            const unassignedTracked = [...trackedFacesRef.current];

            for (const face of currentFaces) {
              let bestMatchIdx = -1;
              let minDistance = 0.3; // 30% of screen distance threshold

              for (let i = 0; i < unassignedTracked.length; i++) {
                const t = unassignedTracked[i];
                const dist = Math.sqrt(Math.pow(face.x - t.x, 2) + Math.pow(face.y - t.y, 2));
                if (dist < minDistance) {
                  minDistance = dist;
                  bestMatchIdx = i;
                }
              }

              if (bestMatchIdx !== -1) {
                const matchedTracked = unassignedTracked.splice(bestMatchIdx, 1)[0];
                matchedFaces.push({ ...face, id: matchedTracked.id });
                const tIdx = trackedFacesRef.current.findIndex(t => t.id === matchedTracked.id);
                if (tIdx !== -1) {
                  trackedFacesRef.current[tIdx].x = face.x;
                  trackedFacesRef.current[tIdx].y = face.y;
                  trackedFacesRef.current[tIdx].lastSeen = now;
                }
              } else {
                const newId = nextFaceIdRef.current++;
                matchedFaces.push({ ...face, id: newId });
                trackedFacesRef.current.push({ id: newId, x: face.x, y: face.y, lastSeen: now });
              }
            }

            // Cleanup old tracked faces
            trackedFacesRef.current = trackedFacesRef.current.filter(t => {
              if (now - t.lastSeen > 1000) {
                delete smoothedScoresRef.current[t.id];
                delete smoothedMoodValueRef.current[t.id];
                delete smoothedHUDRef.current[t.id];
                delete latestBlendshapesRef.current[t.id];
                delete baseValuesRef.current[t.id];
                return false;
              }
              return true;
            });

            for (let i = 0; i < matchedFaces.length; i++) {
              const face = matchedFaces[i];
              const blendshapes = face.blendshapes;
              const landmarks = face.landmarks;
              const faceId = face.id;
              
              const { smoothed, moodTarget } = processEmotions(blendshapes, faceId);
              totalMoodTarget += moodTarget;

              for (const key in smoothed) {
                totalSmoothed[key] += smoothed[key];
              }

              if (showLandmarksRef.current) {
                const drawingUtils = new DrawingUtils(ctx);
                drawingUtils.drawConnectors(
                  landmarks,
                  FaceLandmarker.FACE_LANDMARKS_TESSELATION,
                  { color: "#FFFFFF40", lineWidth: 1 }
                );
                drawingUtils.drawConnectors(
                  landmarks,
                  FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
                  { color: "#FFFFFF", lineWidth: 2 }
                );
                drawingUtils.drawConnectors(
                  landmarks,
                  FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
                  { color: "#FFFFFF", lineWidth: 2 }
                );
                drawingUtils.drawConnectors(
                  landmarks,
                  FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
                  { color: "#FFFFFF", lineWidth: 2 }
                );
                drawingUtils.drawConnectors(
                  landmarks,
                  FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
                  { color: "#FFFFFF", lineWidth: 2 }
                );
                drawingUtils.drawConnectors(
                  landmarks,
                  FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
                  { color: "#FFFFFF", lineWidth: 2 }
                );
                drawingUtils.drawConnectors(
                  landmarks,
                  FaceLandmarker.FACE_LANDMARKS_LIPS,
                  { color: "#FFFFFF", lineWidth: 2 }
                );
              }

              drawHeadMoodMeter(landmarks, ctx, canvas.width, canvas.height, faceId, moodTarget);
            }

            // Update global state based on average mood
            const avgMoodTarget = totalMoodTarget / numFaces;
            vibeValueRef.current = avgMoodTarget;
            
            const avgSmoothed: Record<string, number> = {};
            let avgDom = 'Neutral';
            let maxScore = 0.15;
            
            for (const key in totalSmoothed) {
              avgSmoothed[key] = totalSmoothed[key] / numFaces;
              if (key === 'Sad') continue;
              if (avgSmoothed[key] > maxScore) {
                maxScore = avgSmoothed[key];
                avgDom = key;
              }
            }
            
            if (Math.random() < 0.1 || avgDom !== lastDominantEmotionRef.current) {
              setEmotions(avgSmoothed);
              setVibeValue(avgMoodTarget); // Use average vibe for the global slider
              if (avgDom !== lastDominantEmotionRef.current) {
                setDominantEmotion(avgDom);
                lastDominantEmotionRef.current = avgDom;
              }
            }
          } else {
            if (numFacesDetectedRef.current !== 0) {
              numFacesDetectedRef.current = 0;
              setNumFacesDetected(0);
            }
          }
        } catch (e) {
          // Ignore occasional detection errors
        }
      }
    }
    
    if (webcamRunningRef.current) {
      requestRef.current = requestAnimationFrame(predictWebcam);
    }
  };

  const processEmotions = (blendshapes: any[], faceIndex: number) => {
    latestBlendshapesRef.current[faceIndex] = blendshapes;
    
    const getS = (n: string) => blendshapes.find((b) => b.categoryName === n)?.score || 0;
    
    const currentRaw = {
      biu: getS('browInnerUp'),
      bd: (getS('browDownLeft') + getS('browDownRight')) / 2,
      bid: (getS('browInnerDownLeft') + getS('browInnerDownRight')) / 2,
      ns: (getS('noseSneerLeft') + getS('noseSneerRight')) / 2,
    };

    const base = baseValuesRef.current[faceIndex] || { biu: 0, bd: 0, bid: 0, ns: 0 };
    const nBiu = Math.max(0, currentRaw.biu - base.biu);
    const nBd = Math.max(0, currentRaw.bd - base.bd);
    const nBid = Math.max(0, currentRaw.bid - base.bid);
    const nNs = Math.max(0, currentRaw.ns - base.ns);

    const jo = getS('jawOpen');
    const mf = (getS('mouthFrownLeft') + getS('mouthFrownRight')) / 2;
    const ew = (getS('eyeWideLeft') + getS('eyeWideRight')) / 2;
    const sm = (getS('mouthSmileLeft') + getS('mouthSmileRight')) / 2;
    const eb = (getS('eyeBlinkLeft') + getS('eyeBlinkRight')) / 2;
    const sq = (getS('eyeSquintLeft') + getS('eyeSquintRight')) / 2;
    const ms = (getS('mouthStretchLeft') + getS('mouthStretchRight')) / 2;
    const bou = (getS('browOuterUpLeft') + getS('browOuterUpRight')) / 2;

    // Anger: Brow down + Squint + Nose sneer
    // We use a small threshold (0.03) to ignore resting face noise, 
    // then multiply heavily so a deliberate furrow registers smoothly but strongly.
    // We check both outer (nBd) and inner (nBid) brow down movements.
    let angerBase = 0;
    const maxBrowDown = Math.max(nBd, nBid);
    if (maxBrowDown > 0.03) {
      angerBase = (maxBrowDown - 0.03) * 4.0 + sq * 0.5 + nNs * 0.2;
    }

    let laughBase = sm > 0.4 && jo > 0.15 ? Math.min(1.0, sm * 0.6 + jo * 0.8) : 0;
    let smileBase = laughBase > 0.5 ? 0.2 : sm;

    // Sadness: Brow inner up + Mouth frown. 
    // Reduce sensitivity to browInnerUp alone to avoid false positives.
    const sadBase = (nBiu * 0.5 + mf * 0.5) * (1 - jo * 1.2);

    const targets: Record<string, number> = {
      Smile: smileBase,
      Laugh: laughBase,
      Sad: Math.max(0, sadBase),
      Angry: Math.max(0, angerBase),
      Surprised: jo * 0.8 + ew * 0.2,
      Fear: Math.max(0, (nBiu * 0.4 + bou * 0.3 + ew * 0.3 + ms * 0.1) - 0.15),
      Sleepy: Math.max(0, (eb - 0.6) * 2.5), // Only trigger when eyes are heavily closed to distinguish from Neutral
    };

    if (!smoothedScoresRef.current[faceIndex]) {
      smoothedScoresRef.current[faceIndex] = { Smile: 0, Laugh: 0, Sad: 0, Angry: 0, Surprised: 0, Fear: 0, Sleepy: 0 };
    }
    const smoothed = smoothedScoresRef.current[faceIndex];
    for (const key in smoothed) {
      (smoothed as any)[key] = (smoothed as any)[key] * 0.8 + targets[key] * 0.2;
    }

    let dom = 'Neutral';
    let max = 0.15;
    for (const [n, s] of Object.entries(smoothed)) {
      if (n === 'Sad') continue;
      const score = s as number;
      if (score > max) {
        max = score;
        dom = n;
      }
    }

    // VIBE CHECK calculation logic
    // Use a threshold-based sum to ignore minor noise but still combine strong emotions
    // Multipliers are increased to make the slider more responsive and reach the edges easier.
    const posScore = (smoothed.Smile > 0.05 ? smoothed.Smile * 2.0 : 0) + 
                     (smoothed.Laugh > 0.05 ? smoothed.Laugh * 2.5 : 0) + 
                     (smoothed.Surprised > 0.15 ? smoothed.Surprised * 0.5 : 0);
                     
    const negScore = (smoothed.Angry > 0.05 ? smoothed.Angry * 2.0 : 0) + 
                     (smoothed.Sad > 0.05 ? smoothed.Sad * 1.5 : 0) + 
                     (smoothed.Fear > 0.15 ? smoothed.Fear * 1.2 : 0);
    
    // Calculate target between 0 and 1
    let moodTarget = Math.max(0, Math.min(1, (posScore - negScore + 1) / 2));
    
    // If dominant emotion is Neutral, pull the target to the middle (0.5)
    // so the slider drifts back to center when no strong emotion is detected
    if (dom === 'Neutral') {
      moodTarget = 0.5;
    }

    return { smoothed, moodTarget, dom };
  };

  const drawHeadMoodMeter = (landmarks: any[], ctx: CanvasRenderingContext2D, width: number, height: number, faceIndex: number, moodTarget: number) => {
    const f1 = landmarks[10];
    const lEye = landmarks[33];
    const rEye = landmarks[263];

    const eyeDist = Math.sqrt(Math.pow(rEye.x - lEye.x, 2) + Math.pow(rEye.y - lEye.y, 2));
    const rawScale = eyeDist / 0.16;

    const targetX = f1.x * width;
    const targetY = f1.y * height - 170 * rawScale;
    const rawRoll = Math.atan2(rEye.y - lEye.y, rEye.x - lEye.x);

    if (!smoothedHUDRef.current[faceIndex]) {
      smoothedHUDRef.current[faceIndex] = { x: targetX, y: targetY, roll: rawRoll, scale: rawScale };
    }
    if (smoothedMoodValueRef.current[faceIndex] === undefined) {
      smoothedMoodValueRef.current[faceIndex] = 0.5;
    }

    const hud = smoothedHUDRef.current[faceIndex];
    if (Math.abs(targetX - hud.x) > DEADZONE) hud.x = hud.x * (1 - LERP_VAL) + targetX * LERP_VAL;
    if (Math.abs(targetY - hud.y) > DEADZONE) hud.y = hud.y * (1 - LERP_VAL) + targetY * LERP_VAL;
    hud.roll = hud.roll * (1 - LERP_VAL) + rawRoll * LERP_VAL;
    hud.scale = hud.scale * (1 - LERP_VAL) + rawScale * LERP_VAL;

    smoothedMoodValueRef.current[faceIndex] = smoothedMoodValueRef.current[faceIndex] * 0.90 + moodTarget * 0.10;

    const x = Math.round(hud.x);
    const y = Math.round(hud.y);
    const roll = Math.round(hud.roll * 50) / 50;
    const scale = hud.scale;

    const baseW = 180,
      baseH = 30,
      halfW = baseW / 2;

    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(roll);
    ctx.scale(-scale, scale);

    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.roundRect(-halfW - 2, -2, baseW + 4, baseH + 4, 10);
    ctx.fill();

    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.beginPath();
    ctx.roundRect(-halfW + 6, 6, baseW, baseH, 8);
    ctx.fill();

    const colors = ['#ef4444', '#facc15', '#22c55e'];
    for (let i = 0; i < 3; i++) {
      ctx.fillStyle = colors[i];
      ctx.beginPath();
      const sx = -halfW + i * (baseW / 3);
      const r = i === 0 ? { tl: 8, bl: 8 } : i === 2 ? { tr: 8, br: 8 } : 0;
      ctx.roundRect(sx, 0, baseW / 3, baseH, r as any);
      ctx.fill();
    }

    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.roundRect(-halfW, 0, baseW, baseH, 8);
    ctx.stroke();

    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(-halfW + baseW / 3, 0);
    ctx.lineTo(-halfW + baseW / 3, baseH);
    ctx.moveTo(-halfW + (baseW / 3) * 2, 0);
    ctx.lineTo(-halfW + (baseW / 3) * 2, baseH);
    ctx.stroke();

    const pX = Math.round(-halfW + smoothedMoodValueRef.current[faceIndex] * baseW);
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(pX, baseH + 5);
    ctx.lineTo(pX - 14, baseH + 22);
    ctx.lineTo(pX + 14, baseH + 22);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 4;
    ctx.font = '900 16px "Arial Black", sans-serif';
    ctx.textAlign = 'center';
    ctx.strokeText('VIBE CHECK', 0, -15);
    ctx.fillText('VIBE CHECK', 0, -15);

    ctx.restore();
  };

  const toggleRecording = () => {
    if (isRecording) {
      setIsRecording(false);
      isRecordingRef.current = false;
    } else {
      recentBlendshapesRef.current = [];
      setIsRecording(true);
      isRecordingRef.current = true;
      setGeminiAnalysis(null);
      setShowDetails(false);
    }
  };

  const analyzeWithGemini = async () => {
    if (isRecording) {
      setIsRecording(false);
      isRecordingRef.current = false;
    }

    if (recentBlendshapesRef.current.length === 0) {
      setGeminiAnalysis({ summary: "No face data collected yet. Please start recording to gather data.", details: "" });
      return;
    }

    setIsAnalyzing(true);
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      // Calculate average blendshapes over the recent frames
      const frames = recentBlendshapesRef.current;
      const avgBlendshapes: Record<string, number> = {};
      
      if (frames.length > 0) {
        frames[0].forEach((b: any) => {
          avgBlendshapes[b.categoryName] = 0;
        });
        
        frames.forEach(frame => {
          frame.forEach((b: any) => {
            avgBlendshapes[b.categoryName] += b.score;
          });
        });
        
        Object.keys(avgBlendshapes).forEach(key => {
          avgBlendshapes[key] /= frames.length;
        });
      }

      const prompt = `
You are an AI assistant helping to analyze a audience's engagement and emotional state during the current session.
Here are the average facial blendshape scores (from 0.0 to 1.0) of the student(s) over the recorded time interval. If there are multiple people, this represents the average group emotion:
${JSON.stringify(avgBlendshapes, null, 2)}

Current dominant emotion detected by heuristic model: ${dominantEmotion}
Current Vibe Check (0 = negative, 1 = positive): ${vibeValue.toFixed(2)}
Number of people detected: ${numFacesDetectedRef.current}

Based on these 52 facial muscle activations and the heuristic data, provide a short, insightful summary (1-2 sentences) of how the student(s) are feeling. Then, provide a detailed step-by-step analysis and evidence based on the facial blendshapes to support your summary.
`;

      const response = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: prompt,
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              summary: {
                type: Type.STRING,
                description: "A 1-2 sentence summary of the audience's engagement and emotional state."
              },
              details: {
                type: Type.STRING,
                description: "Detailed analysis steps and evidence based on the facial blendshapes."
              }
            },
            required: ["summary", "details"]
          }
        }
      });

      try {
        const result = JSON.parse(response.text || "{}");
        setGeminiAnalysis({
          summary: result.summary || "No summary generated.",
          details: result.details || ""
        });
      } catch (e) {
        setGeminiAnalysis({ summary: "Failed to parse analysis results.", details: "" });
      }
    } catch (error) {
      console.error("Gemini Analysis Error:", error);
      setGeminiAnalysis({ summary: "Failed to generate analysis. Please check your API key and network connection.", details: "" });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const emojiMap: Record<string, string> = {
    Neutral: '😐',
    Smile: '😊',
    Laugh: '😆',
    Angry: '😠',
    Surprised: '😲',
    Fear: '😨',
    Sleepy: '😴',
  };

  return (
    <div className="w-full max-w-7xl space-y-8 animate-in fade-in duration-500">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
        {/* Video Center */}
        <div className="lg:col-span-2 space-y-6">
          <div className="relative w-full shadow-2xl rounded-3xl overflow-hidden bg-slate-800 border-4 border-slate-700 aspect-video">
            {isModelLoading && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/80 z-20">
                <RefreshCw className="w-10 h-10 text-indigo-500 animate-spin mb-4" />
                <p className="font-bold uppercase tracking-widest text-sm">Loading Neural Engine...</p>
              </div>
            )}
            
            {!webcamRunning && !isModelLoading && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/80 z-20 p-6 text-center">
                {cameraError ? (
                  <>
                    <AlertCircle className="w-16 h-16 text-rose-500 mb-4" />
                    <p className="font-bold uppercase tracking-widest text-sm text-rose-400 mb-2">Camera Error</p>
                    <p className="text-xs text-slate-400 max-w-xs">{cameraError}</p>
                    <p className="text-xs text-slate-500 mt-2 max-w-xs mb-4">Please allow camera access in your browser settings and try again.</p>
                    <button
                      onClick={startCamera}
                      className="bg-slate-700 hover:bg-slate-600 border-2 border-black px-4 py-2 rounded-xl font-black text-white shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] transition-all active:translate-y-1 active:shadow-none uppercase text-xs focus:outline-none focus:ring-4 focus:ring-indigo-500"
                    >
                      Try Again
                    </button>
                  </>
                ) : (
                  <>
                    <Camera className="w-16 h-16 text-slate-600 mb-4" />
                    <p className="font-bold uppercase tracking-widest text-sm text-slate-400">Camera Offline</p>
                  </>
                )}
              </div>
            )}

            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="absolute inset-0 w-full h-full object-cover -scale-x-100"
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full object-cover -scale-x-100 pointer-events-none"
              style={{ imageRendering: 'auto' }}
            />

            {/* Emoji Overlay */}
            {webcamRunning && numFacesDetected === 1 && (
              <div className="absolute top-2 right-2 sm:top-4 sm:right-4 bg-yellow-400 p-4 rounded-2xl border-4 border-black flex flex-col items-center min-w-[120px] z-10 shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] transition-transform scale-[0.3] sm:scale-100 origin-top-right">
                <span className="text-5xl mb-1 transform scale-125 transition-transform duration-200">
                  {emojiMap[dominantEmotion] || '😐'}
                </span>
                <span className="text-xs font-black uppercase tracking-widest text-black">
                  {dominantEmotion}
                </span>
              </div>
            )}
          </div>

          <div className="flex flex-col sm:flex-row justify-center gap-4 w-full">
            <button
              onClick={toggleCamera}
              disabled={isModelLoading}
              className={`flex items-center justify-center w-full sm:w-auto border-4 border-black px-6 py-3 sm:px-8 sm:py-4 rounded-2xl font-black text-white shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] transition-all active:translate-y-1 active:shadow-none uppercase text-xs sm:text-sm whitespace-nowrap focus:outline-none focus:ring-4 focus:ring-indigo-500 ${
                webcamRunning ? 'bg-rose-600 hover:bg-rose-500' : 'bg-indigo-600 hover:bg-indigo-500'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {webcamRunning ? 'Disable Camera' : 'Enable Camera'}
            </button>
            <button
              onClick={calibrateBaseline}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  calibrateBaseline();
                }
              }}
              disabled={!webcamRunning}
              className="flex items-center justify-center w-full sm:w-auto bg-slate-700 hover:bg-slate-600 border-4 border-black px-6 py-3 sm:px-8 sm:py-4 rounded-2xl font-black text-white shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] transition-all active:translate-y-1 active:shadow-none uppercase text-xs sm:text-sm whitespace-nowrap disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-4 focus:ring-indigo-500"
            >
              🎯 Reset Baseline
            </button>
            <button
              onClick={() => {
                const newVal = !showLandmarks;
                setShowLandmarks(newVal);
                showLandmarksRef.current = newVal;
              }}
              disabled={!webcamRunning}
              className="flex items-center justify-center w-full sm:w-auto bg-slate-700 hover:bg-slate-600 border-4 border-black px-6 py-3 sm:px-8 sm:py-4 rounded-2xl font-black text-white shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] transition-all active:translate-y-1 active:shadow-none uppercase text-xs sm:text-sm whitespace-nowrap disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-4 focus:ring-indigo-500"
            >
              {showLandmarks ? 'Hide Landmarks' : 'Show Landmarks'}
            </button>
          </div>
        </div>

        {/* Expanded Metrics Sidebar */}
        <div className="space-y-6">
          <div className="bg-slate-800/80 border-4 border-black p-6 rounded-[2.5rem] backdrop-blur-md w-full shadow-[10px_10px_0px_0px_rgba(0,0,0,1)] text-white">
            <h2 className="text-2xl font-black mb-6 border-b-4 border-black pb-3 flex justify-between items-center uppercase italic">
              {numFacesDetected > 1 ? `Analytics (Avg of ${numFacesDetected})` : 'Analytics'}
              {webcamRunning && (
                <span className="text-[10px] bg-red-500 text-white px-3 py-1 rounded-full border-2 border-black animate-pulse">
                  LIVE
                </span>
              )}
            </h2>

            <div className="space-y-4">
              {/* Group 1: Happiness */}
              <div className="space-y-2 border-b border-slate-700/50 pb-3">
                <MetricBar label="Smile 😊" value={emotions.Smile} color="bg-emerald-400" textColor="text-emerald-400" />
                <MetricBar label="Laugh 😆" value={emotions.Laugh} color="bg-yellow-400" textColor="text-yellow-400" />
              </div>

              {/* Group 2: Reaction */}
              <div className="space-y-2 border-b border-slate-700/50 pb-3">
                <MetricBar label="Surprise 😲" value={emotions.Surprised} color="bg-blue-400" textColor="text-blue-400" />
                <MetricBar label="Fear 😨" value={emotions.Fear} color="bg-purple-400" textColor="text-purple-400" />
              </div>

              {/* Group 3: Negative/Tense */}
              <div className="space-y-2 border-b border-slate-700/50 pb-3">
                <MetricBar label="Anger 😠" value={emotions.Angry} color="bg-rose-500" textColor="text-rose-500" />
              </div>

              {/* Group 4: Sub-metrics */}
              <div className="space-y-2 opacity-60 pt-1">
                <MetricBar label="Sobbing 😭" value={emotions.Sad} color="bg-indigo-500" textColor="text-slate-400" height="h-2" />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Vibe Check Timeline */}
      <div className="bg-slate-800/80 border-4 border-black p-4 rounded-[2.5rem] backdrop-blur-md w-full shadow-[10px_10px_0px_0px_rgba(0,0,0,1)] text-white mb-8">
        <div className="flex justify-between items-center mb-4 border-b-4 border-black pb-3">
          <h2 className="text-xl font-black uppercase italic flex items-center gap-2">
            Vibe Check Timeline
          </h2>
          <select
            value={timeWindow}
            onChange={(e) => setTimeWindow(Number(e.target.value))}
            className="bg-slate-900 border-2 border-slate-600 rounded-xl px-3 py-1 text-xs font-bold outline-none cursor-pointer focus:ring-2 focus:ring-indigo-500"
          >
            <option value={15}>15s Window</option>
            <option value={30}>30s Window</option>
            <option value={60}>1m Window</option>
            <option value={300}>5m Window</option>
          </select>
        </div>
        <div className="h-[180px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={vibeHistory} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
              <XAxis 
                dataKey="time" 
                type="number" 
                domain={['dataMin', 'dataMax']} 
                tickFormatter={(unixTime) => new Date(unixTime).toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' })}
                stroke="#94a3b8" 
                tick={{ fontSize: 10 }}
                minTickGap={30}
              />
              <YAxis 
                domain={[0, 1]} 
                stroke="#94a3b8" 
                tick={{ fontSize: 10 }}
                ticks={[0, 0.25, 0.5, 0.75, 1]}
                tickFormatter={(val) => val === 0 ? 'Neg' : val === 1 ? 'Pos' : val.toFixed(2)}
              />
              <Tooltip 
                labelFormatter={(label) => new Date(label).toLocaleTimeString()}
                formatter={(value: number) => [value.toFixed(2), 'Vibe Score']}
                contentStyle={{ backgroundColor: '#0f172a', border: '2px solid #000', borderRadius: '8px', fontWeight: 'bold' }}
              />
              <Line 
                type="monotone" 
                dataKey="score" 
                stroke="#34d399" 
                strokeWidth={3} 
                dot={false} 
                isAnimationActive={false}
                connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Gemini Analysis Panel */}
      <div className="bg-indigo-950/80 border-4 border-indigo-500 p-4 rounded-[2.5rem] backdrop-blur-md w-full shadow-[10px_10px_0px_0px_rgba(99,102,241,0.4)] text-white">
        <h2 className="text-lg font-black mb-4 border-b-4 border-indigo-500/50 pb-2 flex justify-between items-center uppercase italic text-indigo-300">
          AI Professor Assistant
          <Activity className="w-5 h-5 text-indigo-400" />
        </h2>

        <div className="flex flex-col lg:flex-row gap-6">
          {/* Left Side: Controls */}
          <div className="w-full lg:w-1/3 space-y-3">
            {numFacesDetected > 1 && (
              <div className="bg-emerald-500/20 text-emerald-300 border border-emerald-500/50 p-2 rounded-xl text-xs font-bold flex items-center gap-2">
                <AudioLines className="w-4 h-4 shrink-0" />
                Analyzing average group emotion
              </div>
            )}

            <div className="space-y-1">
              <label className="text-[10px] font-black uppercase tracking-widest text-indigo-400 ml-1">
                Sampling Frequency
              </label>
              <div className="flex gap-2">
                <select
                  value={recordFrequency}
                  onChange={(e) => {
                    const val = Number(e.target.value);
                    setRecordFrequency(val);
                    recordFrequencyRef.current = val;
                  }}
                  className="bg-indigo-900 border-2 border-indigo-500 rounded-xl px-2 py-1.5 text-xs font-bold outline-none cursor-pointer flex-1 focus:ring-2 focus:ring-indigo-300"
                >
                  <option value={100}>10 fps</option>
                  <option value={200}>5 fps</option>
                  <option value={500}>2 fps</option>
                  <option value={1000}>1 fps</option>
                </select>

                <button
                  onClick={toggleRecording}
                  disabled={!webcamRunning || isAnalyzing}
                  className={`flex-1 border-2 border-black px-3 py-1.5 rounded-xl font-black text-white shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] transition-all active:translate-y-1 active:shadow-none uppercase text-[10px] flex items-center justify-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-4 focus:ring-indigo-500 ${
                    isRecording ? 'bg-rose-500 hover:bg-rose-400' : 'bg-emerald-500 hover:bg-emerald-400'
                  }`}
                >
                  {isRecording ? (
                    <>
                      <div className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                      Stop
                    </>
                  ) : (
                    <>
                      <div className="w-1.5 h-1.5 rounded-full bg-white" />
                      Record
                    </>
                  )}
                </button>
              </div>
            </div>

            <button
              onClick={analyzeWithGemini}
              disabled={isAnalyzing || (!isRecording && recentBlendshapesRef.current.length === 0)}
              className="w-full bg-indigo-500 hover:bg-indigo-400 border-4 border-black px-4 py-3 rounded-2xl font-black text-white shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] transition-all active:translate-y-1 active:shadow-none uppercase text-xs flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-4 focus:ring-indigo-500"
            >
              {isAnalyzing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <AudioLines className="w-4 h-4" />
                  Give Analysis Result
                </>
              )}
            </button>
          </div>

          {/* Right Side: Content Window */}
          <div className="flex-1 min-h-[150px]">
            {geminiAnalysis ? (
              <div className="text-sm leading-relaxed text-indigo-100 bg-indigo-900/50 p-4 rounded-3xl border border-indigo-500/30 flex flex-col gap-3 h-full shadow-inner">
                <div className="flex items-center gap-2 text-indigo-300 font-black uppercase text-[10px] tracking-widest border-b border-indigo-500/20 pb-1">
                  <Activity className="w-3 h-3" />
                  Executive Summary
                </div>
                <p className="font-medium text-base text-white leading-snug">{geminiAnalysis.summary}</p>

                {geminiAnalysis.details && (
                  <div className="mt-auto pt-2">
                    <button
                      onClick={() => setShowDetails(!showDetails)}
                      className="flex items-center gap-1 text-[10px] font-black uppercase tracking-widest text-indigo-400 hover:text-indigo-300 transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 rounded px-1"
                    >
                      {showDetails ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                      {showDetails ? "Hide Technical Breakdown" : "Show Technical Breakdown"}
                    </button>

                    {showDetails && (
                      <div className="mt-2 p-3 bg-black/30 rounded-xl border border-indigo-500/10 whitespace-pre-wrap text-indigo-200/80 text-[10px] font-mono leading-relaxed max-h-[200px] overflow-y-auto custom-scrollbar">
                        {geminiAnalysis.details}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="h-full min-h-[150px] text-xs text-indigo-300/70 italic flex flex-col items-center justify-center gap-3 bg-indigo-900/20 p-6 rounded-3xl border-2 border-dashed border-indigo-500/20 text-center">
                <div className="p-3 bg-indigo-500/10 rounded-full">
                  <AlertCircle className="w-6 h-6 text-indigo-500/50" />
                </div>
                <p className="max-w-xs">
                  Awaiting camera feed. Start recording to capture emotional metrics, then request an AI analysis.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricBar({ label, value, color, textColor, height = "h-3" }: { label: string, value: number, color: string, textColor: string, height?: string }) {
  const percent = Math.min(Math.round(value * 100), 100);
  return (
    <div>
      <div className={`flex justify-between font-black uppercase text-[10px] tracking-wider mb-1 ${textColor}`}>
        <span>{label}</span>
        <span>{percent}%</span>
      </div>
      <div className={`w-full bg-black rounded-full border border-black overflow-hidden ${height}`}>
        <div
          className={`h-full ${color} transition-all duration-100 ease-out`}
          style={{ width: `${percent}%` }}
        ></div>
      </div>
    </div>
  );
}
