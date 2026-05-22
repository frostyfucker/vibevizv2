import React, { useEffect, useRef, useState } from 'react';
import { HandLandmarker, FaceLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';
import { GoogleGenAI, Type } from '@google/genai';
import { Activity, Camera, RefreshCw, AlertCircle, ChevronDown, ChevronUp, Hand, Play, Pause, Square } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const targetBlendshapes = [
  'mouthPucker', 'mouthFunnel', 'mouthSmileLeft', 'mouthSmileRight', 'jawOpen'
];

function normalizeHand(landmarks: any[]) {
  if (!landmarks || landmarks.length === 0) return null;
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  
  const scale = Math.sqrt(
    Math.pow(middleMcp.x - wrist.x, 2) + 
    Math.pow(middleMcp.y - wrist.y, 2) + 
    Math.pow(middleMcp.z - wrist.z, 2)
  ) || 1;

  return landmarks.map(lm => ({
    x: (lm.x - wrist.x) / scale,
    y: (lm.y - wrist.y) / scale,
    z: (lm.z - wrist.z) / scale,
  }));
}

function getFingerStates(landmarks: any[]) {
  if (!landmarks || landmarks.length < 21) return [false, false, false, false, false];
  const dist = (p1: any, p2: any) => Math.sqrt(Math.pow(p1.x-p2.x,2) + Math.pow(p1.y-p2.y,2) + Math.pow(p1.z-p2.z,2));
  const wrist = landmarks[0];
  return [
    dist(wrist, landmarks[4]) > dist(wrist, landmarks[3]),
    dist(wrist, landmarks[8]) > dist(wrist, landmarks[6]),
    dist(wrist, landmarks[12]) > dist(wrist, landmarks[10]),
    dist(wrist, landmarks[16]) > dist(wrist, landmarks[14]),
    dist(wrist, landmarks[20]) > dist(wrist, landmarks[18])
  ];
}

function extractBlendshapes(blendshapes: any[]) {
  if (!blendshapes || blendshapes.length === 0) return {};
  const categories = blendshapes[0].categories;
  const sorted = [...categories].sort((a, b) => b.score - a.score);
  const top3 = sorted.slice(0, 3);
  const result: Record<string, number> = {};
  for (const cat of top3) {
    if (cat.score > 0.1) {
      result[cat.categoryName] = parseFloat(cat.score.toFixed(3));
    }
  }
  return result;
}

export default function SignLanguageAnalyticsTab() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [handLandmarker, setHandLandmarker] = useState<HandLandmarker | null>(null);
  const [faceLandmarker, setFaceLandmarker] = useState<FaceLandmarker | null>(null);
  const [mediaSource, setMediaSource] = useState<'none' | 'webcam' | 'video'>('none');
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const isPlayingRef = useRef(false);
  const [videoProgress, setVideoProgress] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [isHoveringVideo, setIsHoveringVideo] = useState(false);
  const hoverTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleMouseMove = () => {
    setIsHoveringVideo(true);
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
    }
    hoverTimeoutRef.current = setTimeout(() => {
      if (isPlayingRef.current) {
        setIsHoveringVideo(false);
      }
    }, 2500);
  };

  const handleMouseLeave = () => {
    if (isPlayingRef.current) {
      setIsHoveringVideo(false);
    }
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
    }
  };
  const [isModelLoading, setIsModelLoading] = useState(true);

  const [handActivity, setHandActivity] = useState(0);
  const handActivityRef = useRef<number>(0);
  const [faceActivity, setFaceActivity] = useState(0);
  const faceActivityRef = useRef<number>(0);

  const [timeWindow, setTimeWindow] = useState<number>(15);
  const timeWindowRef = useRef<number>(15);
  
  // Update ref when state changes
  useEffect(() => {
    timeWindowRef.current = timeWindow;
  }, [timeWindow]);

  const [activityHistory, setActivityHistory] = useState<{time: number, score: number | null}[]>([]);

  interface AnalysisResult {
    summary: string;
    details: string;
  }

  const [geminiAnalysis, setGeminiAnalysis] = useState<AnalysisResult | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const [detectedEmoji, setDetectedEmoji] = useState<string | null>(null);
  const detectedEmojiRef = useRef<string | null>(null);

  const [interactiveAnimation, setInteractiveAnimation] = useState(false);
  const [popups, setPopups] = useState<{id: number, emoji: string, left: number, size: number, duration: number, tx: number, createdAt: number}[]>([]);

  useEffect(() => {
    if (!interactiveAnimation || !detectedEmoji) return;
    
    const targetEmojis = ['👌', '🫶', '🤫', '✌️', '👍'];
    if (!targetEmojis.includes(detectedEmoji)) return;

    const displayEmoji = detectedEmoji === '🫶' ? '💗' : detectedEmoji;

    const interval = setInterval(() => {
      setPopups(prev => {
        const newPopups = [];
        for (let i = 0; i < 4; i++) {
          newPopups.push({
            id: Date.now() + i + Math.random(),
            emoji: displayEmoji,
            left: Math.random() * 100, // %
            size: Math.random() * 3 + 2, // rem
            duration: Math.random() * 2 + 2, // seconds
            tx: (Math.random() - 0.5) * 200, // translation X
            createdAt: Date.now()
          });
        }
        return [...prev.slice(-50), ...newPopups];
      });
    }, 150);

    return () => clearInterval(interval);
  }, [detectedEmoji, interactiveAnimation]);

  useEffect(() => {
    const cleanup = setInterval(() => {
      setPopups(prev => prev.filter(p => Date.now() - p.createdAt < p.duration * 1000));
    }, 1000);
    return () => clearInterval(cleanup);
  }, []);

  const [isRecording, setIsRecording] = useState(false);
  const isRecordingRef = useRef(false);
  const [showLandmarks, setShowLandmarks] = useState(false);
  const showLandmarksRef = useRef(false);
  const [recordFrequency, setRecordFrequency] = useState(100); // 10fps
  const recordFrequencyRef = useRef(100);
  const lastRecordTimeRef = useRef<number>(0);
  const recordingStartTimeRef = useRef<number>(0);

  const requestRef = useRef<number>(0);
  const lastVideoTimeRef = useRef<number>(-1);
  const recentDataRef = useRef<any[]>([]);
  const lastHandRef = useRef<any[][] | null>(null);
  const lastFaceRef = useRef<any[] | null>(null);
  const drawingUtilsRef = useRef<DrawingUtils | null>(null);

  useEffect(() => {
    let active = true;
    async function setupModel() {
      try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm'
        );
        const hLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numHands: 2,
        });
        const fLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numFaces: 1,
          outputFaceBlendshapes: true,
        });
        if (active) {
          setHandLandmarker(hLandmarker);
          setFaceLandmarker(fLandmarker);
          setIsModelLoading(false);
        } else {
          hLandmarker.close();
          fLandmarker.close();
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

  useEffect(() => {
    return () => {
      if (handLandmarker) handLandmarker.close();
      if (faceLandmarker) faceLandmarker.close();
    };
  }, [handLandmarker, faceLandmarker]);

  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      const now = Date.now();
      const score = (handActivityRef.current + faceActivityRef.current) / 2;
      setActivityHistory(prev => {
        const cutoff = now - timeWindow * 1000;
        const filtered = prev.filter(p => p.time >= cutoff);
        return [...filtered, { time: now, score }];
      });
    }, 200);
    return () => clearInterval(interval);
  }, [isPlaying, timeWindow]);

  const startCamera = async () => {
    if (!handLandmarker || !faceLandmarker) return;
    setCameraError(null);
    try {
      stopMedia();
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setMediaSource('webcam');
          setIsPlaying(true);
          isPlayingRef.current = true;
          if (requestRef.current) cancelAnimationFrame(requestRef.current);
          predictLoop();
        };
      }
    } catch (err: any) {
      console.error('Error accessing webcam', err);
      setCameraError(err.message || 'Permission denied');
    }
  };

  useEffect(() => {
    if (handLandmarker && faceLandmarker && !isPlayingRef.current) {
      startCamera();
    }
  }, [handLandmarker, faceLandmarker]);

  const stopMedia = () => {
    setIsPlaying(false);
    isPlayingRef.current = false;
    setMediaSource('none');
    
    if (videoRef.current) {
      videoRef.current.onloadedmetadata = null;
      if (videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
        videoRef.current.srcObject = null;
      }
      videoRef.current.src = '';
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
    if (mediaSource === 'webcam') {
      stopMedia();
    } else {
      startCamera();
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setGeminiAnalysis(null);
      const oldSrc = videoRef.current?.src;
      stopMedia();
      if (oldSrc && oldSrc.startsWith('blob:')) {
        URL.revokeObjectURL(oldSrc);
      }
      const url = URL.createObjectURL(file);
      if (videoRef.current) {
        videoRef.current.srcObject = null;
        videoRef.current.src = url;
        setMediaSource('video');
        setVideoProgress(0);
        setVideoDuration(0);
      }
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const processFrame = () => {
    if (!videoRef.current || !canvasRef.current || !handLandmarker || !faceLandmarker) return;

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
          const handResults = handLandmarker.detectForVideo(video, performance.now());
          const faceResults = faceLandmarker.detectForVideo(video, performance.now());
          
          ctx.resetTransform();
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          let currentHandActivity = 0;
          let currentFaceActivity = 0;

          if (handResults && handResults.landmarks?.length > 0) {
            let movement = 0;
            if (lastHandRef.current && lastHandRef.current.length === handResults.landmarks.length) {
              for (let h = 0; h < handResults.landmarks.length; h++) {
                for (let i = 0; i < handResults.landmarks[h].length; i++) {
                  movement += Math.abs(handResults.landmarks[h][i].x - lastHandRef.current[h][i].x) + 
                              Math.abs(handResults.landmarks[h][i].y - lastHandRef.current[h][i].y);
                }
              }
            }
            lastHandRef.current = handResults.landmarks;
            currentHandActivity = Math.min(1, movement * 5); // Scale up movement
            
            let currentEmoji = null;
            const hands = handResults.landmarks;
            const face = faceResults?.faceLandmarks?.[0];
            
            if (hands.length === 2) {
               const h1 = hands[0];
               const h2 = hands[1];
               const thumbDist = Math.hypot(h1[4].x - h2[4].x, h1[4].y - h2[4].y);
               const indexDist = Math.hypot(h1[8].x - h2[8].x, h1[8].y - h2[8].y);
               const wristDist = Math.hypot(h1[0].x - h2[0].x, h1[0].y - h2[0].y);
               
               if (thumbDist < 0.1 && indexDist < 0.1) {
                 currentEmoji = '🫶';
               }
            }
            
            if (!currentEmoji && hands.length > 0) {
               for (let i = 0; i < hands.length; i++) {
                 const h = hands[i];
                 const handedness = handResults.handednesses[i][0].categoryName;
                 const dist = (p1: any, p2: any) => Math.hypot(p1.x - p2.x, p1.y - p2.y);
                 
                 const getAngleBetween = (p1: any, p2: any, center: any) => {
                   const v1 = { x: p1.x - center.x, y: p1.y - center.y };
                   const v2 = { x: p2.x - center.x, y: p2.y - center.y };
                   const dot = v1.x * v2.x + v1.y * v2.y;
                   const mag1 = Math.hypot(v1.x, v1.y);
                   const mag2 = Math.hypot(v2.x, v2.y);
                   return Math.acos(dot / (mag1 * mag2)) * (180 / Math.PI);
                 };
                 
                 // Note: Face and Hand models have different relative Z spaces. 
                 // We use a loose Z check (0.3) to prevent extreme false depth matches, 
                 // but rely primarily on 2D distance for robust detection.
                 const isTouching = (p1: any, p2: any, t2d = 0.15, tz = 0.3) => 
                   dist(p1, p2) < t2d && Math.abs((p1.z || 0) - (p2.z || 0)) < tz;
                 
                 const indexUp = dist(h[8], h[0]) > dist(h[6], h[0]);
                 const middleUp = dist(h[12], h[0]) > dist(h[10], h[0]);
                 const ringUp = dist(h[16], h[0]) > dist(h[14], h[0]);
                 const pinkyUp = dist(h[20], h[0]) > dist(h[18], h[0]);
                 
                 const handSize = dist(h[0], h[9]);
                 const thumbExtended = dist(h[4], h[0]) > dist(h[5], h[0]) * 1.1 && dist(h[4], h[5]) > handSize * 0.3;
                 const thumbUp = thumbExtended && h[4].y < h[5].y - 0.05;
                 const thumbIndexDist = dist(h[4], h[8]);

                 const isThumbUpright = h[4].y < h[3].y && h[3].y < h[2].y && Math.abs(h[4].y - h[2].y) > Math.abs(h[4].x - h[2].x);
                 const fingersCurled = !indexUp && !middleUp && !ringUp && !pinkyUp;
                 const thumbHigherThanFingers = h[4].y < Math.min(h[8].y, h[12].y, h[16].y, h[20].y);
                 
                 const rectPoints = [h[6], h[7], h[10], h[11], h[14], h[15], h[18], h[19]];
                 const minRectX = Math.min(...rectPoints.map(p => p.x));
                 const maxRectX = Math.max(...rectPoints.map(p => p.x));
                 const minRectY = Math.min(...rectPoints.map(p => p.y));
                 const maxRectY = Math.max(...rectPoints.map(p => p.y));
                 
                 const padX = (maxRectX - minRectX) * 0.2;
                 const padY = (maxRectY - minRectY) * 0.2;
                 
                 const thumbInRect = 
                   h[4].x >= minRectX - padX && h[4].x <= maxRectX + padX &&
                   h[4].y >= minRectY - padY && h[4].y <= maxRectY + padY;

                 if (face) {
                   const topOfHead = face[10];
                   const leftCheek = face[58]; // lower cheek/beside mouth
                   const rightCheek = face[288];
                   const mouth = face[13]; // upper lip
                   const leftEye = face[159];
                   const rightEye = face[386];
                   const wrist = h[0];
                   const indexTip = h[8];

                   // Keep Silence 🤫: index up, others down, index tip touching mouth
                   // Removed strict verticality check because hand is often tilted when touching mouth
                   if (indexUp && !middleUp && !ringUp && !pinkyUp && isTouching(indexTip, mouth, 0.15, 0.3)) {
                     currentEmoji = '🤫';
                     break;
                   }

                   // Confused 😕: hand touching top of head
                   // Relaxed to just check middle finger to allow for angled hand
                   const fingersTouchingHead = 
                     isTouching(h[12], topOfHead, 0.5, 0.5) && 
                     isTouching(h[9], topOfHead, 0.5, 0.5);

                   const thumbStraight = dist(h[4], h[2]) > dist(h[3], h[2]);
                   const fiveFingersStraight = indexUp && middleUp && ringUp && pinkyUp && thumbStraight;
                   
                   const fiveFingersClose = 
                     // Tips
                     dist(h[4], h[8]) < handSize * 0.7 &&
                     dist(h[8], h[12]) < handSize * 0.4 &&
                     dist(h[12], h[16]) < handSize * 0.4 &&
                     dist(h[16], h[20]) < handSize * 0.4 &&
                     // DIP / IP joints
                     dist(h[3], h[7]) < handSize * 0.7 &&
                     dist(h[7], h[11]) < handSize * 0.4 &&
                     dist(h[11], h[15]) < handSize * 0.4 &&
                     dist(h[15], h[19]) < handSize * 0.4 &&
                     // PIP joints
                     dist(h[2], h[6]) < handSize * 0.7 &&
                     dist(h[6], h[10]) < handSize * 0.4 &&
                     dist(h[10], h[14]) < handSize * 0.4 &&
                     dist(h[14], h[18]) < handSize * 0.4;
                     
                   const vIndex = { x: h[8].x - h[5].x, y: h[8].y - h[5].y };
                   const vMiddle = { x: h[12].x - h[9].x, y: h[12].y - h[9].y };
                   const vRing = { x: h[16].x - h[13].x, y: h[16].y - h[13].y };
                   const vPinky = { x: h[20].x - h[17].x, y: h[20].y - h[17].y };
                   const vThumb = { x: h[4].x - h[2].x, y: h[4].y - h[2].y };
                   
                   const normalize = (v: any) => {
                     const mag = Math.hypot(v.x, v.y) || 1;
                     return { x: v.x / mag, y: v.y / mag };
                   };
                   
                   const nIndex = normalize(vIndex);
                   const nMiddle = normalize(vMiddle);
                   const nRing = normalize(vRing);
                   const nPinky = normalize(vPinky);
                   const nThumb = normalize(vThumb);
                   
                   const dotP = (v1: any, v2: any) => v1.x * v2.x + v1.y * v2.y;
                   
                   const fingersParallel = 
                     dotP(nIndex, nMiddle) > 0.7 &&
                     dotP(nMiddle, nRing) > 0.7 &&
                     dotP(nRing, nPinky) > 0.7 &&
                     dotP(nIndex, nThumb) > 0.5;
                     
                   const eyeY = Math.min(leftEye.y, rightEye.y);
                   // Exclude wrist (h[0]) from the eye check, as it often drops below eyes when angled
                   const allHandAboveEyes = h.slice(1).every((p: any) => p.y < eyeY);
                   
                   const getAngleFromHorizon = (tip: any, wrist: any) => {
                     // Opposite: y-axis distance from finger top to wrist point
                     // (wrist.y is larger than tip.y when hand is up, so wrist.y - tip.y is positive)
                     const opposite = wrist.y - tip.y; 
                     
                     // Adjacent: x-axis distance (absolute value supports both left and right hands)
                     const adjacent = Math.abs(tip.x - wrist.x); 
                     
                     // Hypotenuse: the length of the magenta line
                     const hypotenuse = Math.hypot(adjacent, opposite);
                     
                     // Sine = Opposite / Hypotenuse
                     const sinValue = opposite / hypotenuse;
                     
                     // Convert sine value back to an angle in degrees
                     return Math.asin(sinValue) * (180 / Math.PI);
                   };
                   
                   const angleThumb = getAngleFromHorizon(h[4], h[0]);
                   const angleIndex = getAngleFromHorizon(h[8], h[0]);
                   const angleMiddle = getAngleFromHorizon(h[12], h[0]);
                   const angleRing = getAngleFromHorizon(h[16], h[0]);
                   const anglePinky = getAngleFromHorizon(h[20], h[0]);
                   
                   const is40to85Degrees = (angle: number) => angle >= 40 && angle <= 85;
                   const fingersAngled40to85 = 
                     is40to85Degrees(angleThumb) &&
                     is40to85Degrees(angleIndex) &&
                     is40to85Degrees(angleMiddle) &&
                     is40to85Degrees(angleRing) &&
                     is40to85Degrees(anglePinky);

                   if (fingersTouchingHead && fiveFingersStraight && fiveFingersClose && fingersParallel && allHandAboveEyes && fingersAngled40to85) {
                     currentEmoji = '😕';
                     break;
                   }

                   // Think 🤔: The four fingers except thumbs should close to each other and roughly cover mouth (cross mouth)
                   const fourFingersExtended = indexUp && middleUp && ringUp && pinkyUp;
                   const fourFingersClose = 
                     dist(h[8], h[12]) < handSize * 0.4 &&
                     dist(h[12], h[16]) < handSize * 0.4 &&
                     dist(h[16], h[20]) < handSize * 0.4;
                   
                   const isCoveringMouth = 
                     isTouching(h[12], mouth, 0.2, 0.4) || 
                     isTouching(h[9], mouth, 0.2, 0.4) ||
                     isTouching(h[8], mouth, 0.2, 0.4) ||
                     isTouching(h[5], mouth, 0.2, 0.4);

                   // All points of the hand (including wrist) must be below the eyes
                   const isBelowEyes = h.every((p: any) => p.y > Math.min(leftEye.y, rightEye.y));

                   if (fourFingersExtended && fourFingersClose && isCoveringMouth && isBelowEyes) {
                     currentEmoji = '🤔';
                     break;
                   }
                 }

                 if (thumbIndexDist < 0.08 && middleUp && ringUp && pinkyUp) {
                   currentEmoji = '👌';
                   break;
                 } else if (isThumbUpright && fingersCurled && thumbHigherThanFingers && thumbExtended && !thumbInRect && h[4].y < minRectY) {
                   currentEmoji = '👍';
                   break;
                 } else if (!thumbExtended && indexUp && middleUp && !ringUp && !pinkyUp) {
                   currentEmoji = '✌️';
                   break;
                 } else if (indexUp && middleUp && ringUp && pinkyUp) {
                   const angle1 = getAngleBetween(h[4], h[8], h[0]);
                   const angle2 = getAngleBetween(h[8], h[12], h[0]);
                   const angle3 = getAngleBetween(h[12], h[16], h[0]);
                   const angle4 = getAngleBetween(h[16], h[20], h[0]);

                   const angles = [angle1, angle2, angle3, angle4];
                   const minAngle = Math.min(...angles);
                   const maxAngle = Math.max(...angles);
                   
                   // Angle between contiguous fingers are roughly the same and > 5 degrees
                   const fingersSeparate = minAngle > 5 && (maxAngle - minAngle) < 35;

                   const fingersPointingUp = 
                     h[8].y < h[5].y &&
                     h[12].y < h[9].y &&
                     h[16].y < h[13].y &&
                     h[20].y < h[17].y &&
                     h[4].y < h[1].y;

                   const isHandUpright = h[0].y > h[9].y && Math.abs(h[9].y - h[0].y) > Math.abs(h[9].x - h[0].x);

                   if (fingersSeparate && fingersPointingUp && isHandUpright) {
                     // Prevent "Open" from triggering if hand is covering the face
                     let isCoveringFace = false;
                     if (face) {
                       const nose = face[1];
                       if (dist(h[9], nose) < 0.15) isCoveringFace = true;
                     }
                     if (!isCoveringFace) {
                       currentEmoji = '🖐️';
                       break;
                     }
                   }
                 } else if (fingersCurled && thumbInRect) {
                   currentEmoji = '✊';
                   break;
                 }
               }
            }
            
            if (currentEmoji !== detectedEmojiRef.current) {
               detectedEmojiRef.current = currentEmoji;
               setDetectedEmoji(currentEmoji);
            }

            if (showLandmarksRef.current) {
              if (!drawingUtilsRef.current) {
                drawingUtilsRef.current = new DrawingUtils(ctx);
              }
              const drawingUtils = drawingUtilsRef.current;
              for (const landmarks of handResults.landmarks) {
                drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 3 });
                drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", lineWidth: 2, radius: 4 });
                
                // Draw the fist bounding box for debugging
                const rectPoints = [landmarks[6], landmarks[7], landmarks[10], landmarks[11], landmarks[14], landmarks[15], landmarks[18], landmarks[19]];
                const minRectX = Math.min(...rectPoints.map(p => p.x));
                const maxRectX = Math.max(...rectPoints.map(p => p.x));
                const minRectY = Math.min(...rectPoints.map(p => p.y));
                const maxRectY = Math.max(...rectPoints.map(p => p.y));
                
                const padX = (maxRectX - minRectX) * 0.2;
                const padY = (maxRectY - minRectY) * 0.2;
                
                const x = (minRectX - padX) * canvas.width;
                const y = (minRectY - padY) * canvas.height;
                const w = (maxRectX - minRectX + padX * 2) * canvas.width;
                const h = (maxRectY - minRectY + padY * 2) * canvas.height;
                
                ctx.strokeStyle = '#00FFFF';
                ctx.lineWidth = 2;
                ctx.strokeRect(x, y, w, h);
                
                // Draw lines from fingertips to wrist for debugging Confused gesture angles
                ctx.strokeStyle = '#FF00FF'; // Magenta
                ctx.lineWidth = 2;
                ctx.font = '14px Arial';
                ctx.fillStyle = '#FF00FF';
                const tips = [4, 8, 12, 16, 20];
                const wristPoint = landmarks[0];
                tips.forEach(tipIdx => {
                  const tip = landmarks[tipIdx];
                  ctx.beginPath();
                  ctx.moveTo(wristPoint.x * canvas.width, wristPoint.y * canvas.height);
                  ctx.lineTo(tip.x * canvas.width, tip.y * canvas.height);
                  ctx.stroke();
                  
                  // Calculate angle for debugging display
                  const opposite = wristPoint.y - tip.y; 
                  const adjacent = Math.abs(tip.x - wristPoint.x); 
                  const hypotenuse = Math.hypot(adjacent, opposite);
                  const sinValue = opposite / hypotenuse;
                  const angle = Math.asin(sinValue) * (180 / Math.PI);
                  
                  // Draw the angle text near the fingertip
                  ctx.fillText(`${Math.round(angle)}°`, tip.x * canvas.width + 10, tip.y * canvas.height - 10);
                });
              }
            }
          } else {
            lastHandRef.current = null;
            if (detectedEmojiRef.current !== null) {
               detectedEmojiRef.current = null;
               setDetectedEmoji(null);
            }
          }

          if (faceResults && faceResults.faceLandmarks?.length > 0) {
            const landmarks = faceResults.faceLandmarks[0];
            let movement = 0;
            if (lastFaceRef.current) {
              // Check key expressive points (mouth, eyebrows)
              const keyPoints = [13, 14, 33, 263, 10, 152]; // Lips, eyes, top/bottom
              for (const idx of keyPoints) {
                movement += Math.abs(landmarks[idx].x - lastFaceRef.current[idx].x) + 
                            Math.abs(landmarks[idx].y - lastFaceRef.current[idx].y);
              }
            }
            lastFaceRef.current = landmarks;
            currentFaceActivity = Math.min(1, movement * 20); // Scale up movement
            
            if (showLandmarksRef.current) {
              if (!drawingUtilsRef.current) {
                drawingUtilsRef.current = new DrawingUtils(ctx);
              }
              const drawingUtils = drawingUtilsRef.current;
              drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#C0C0C070", lineWidth: 1 });
              drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#FF3030", lineWidth: 2 });
              drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#FF3030", lineWidth: 2 });
              drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#30FF30", lineWidth: 2 });
              drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: "#30FF30", lineWidth: 2 });
              drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#E0E0E0", lineWidth: 2 });
              drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "#FF3030", lineWidth: 2 });
            }
          } else {
            lastFaceRef.current = null;
          }

          handActivityRef.current = handActivityRef.current * 0.8 + currentHandActivity * 0.2;
          faceActivityRef.current = faceActivityRef.current * 0.8 + currentFaceActivity * 0.2;

          if (Math.random() < 0.1) {
            setHandActivity(handActivityRef.current);
            setFaceActivity(faceActivityRef.current);
          }

          if (isRecordingRef.current) {
            const now = performance.now();
            if (now - lastRecordTimeRef.current >= recordFrequencyRef.current) {
              
              let leftHand = null;
              let rightHand = null;
              let rawLeftWrist = null;
              let rawRightWrist = null;
              let leftFingerStates = null;
              let rightFingerStates = null;
              
              if (handResults && handResults.landmarks) {
                handResults.handednesses.forEach((handednessArray: any[], index: number) => {
                  const category = handednessArray[0].categoryName;
                  const normalized = normalizeHand(handResults.landmarks[index]);
                  const fStates = getFingerStates(handResults.landmarks[index]);
                  if (category === 'Left') {
                    leftHand = normalized;
                    rawLeftWrist = handResults.landmarks[index][0];
                    leftFingerStates = fStates;
                  }
                  if (category === 'Right') {
                    rightHand = normalized;
                    rawRightWrist = handResults.landmarks[index][0];
                    rightFingerStates = fStates;
                  }
                });
              }

              let faceShapes = {};
              let nosePos = null;
              if (faceResults && faceResults.faceBlendshapes) {
                faceShapes = extractBlendshapes(faceResults.faceBlendshapes);
              }
              if (faceResults && faceResults.faceLandmarks && faceResults.faceLandmarks.length > 0) {
                nosePos = faceResults.faceLandmarks[0][4]; // index 4 is nose tip
              }

              let hL_nose = null;
              let hR_nose = null;
              if (nosePos) {
                if (rawLeftWrist) hL_nose = [rawLeftWrist.x - nosePos.x, rawLeftWrist.y - nosePos.y, rawLeftWrist.z - nosePos.z];
                if (rawRightWrist) hR_nose = [rawRightWrist.x - nosePos.x, rawRightWrist.y - nosePos.y, rawRightWrist.z - nosePos.z];
              }

              const currentTime = Math.round(now - recordingStartTimeRef.current);
              recentDataRef.current.push({
                time: currentTime,
                hands: { left: leftHand, right: rightHand },
                leftFingerStates,
                rightFingerStates,
                hL_nose,
                hR_nose,
                face: faceShapes
              });

              const maxFrames = (timeWindowRef.current * 1000) / recordFrequencyRef.current;
              if (recentDataRef.current.length > maxFrames) {
                recentDataRef.current = recentDataRef.current.slice(-maxFrames);
              }

              lastRecordTimeRef.current = now;
            }
          }

        } catch (e) {
          // Ignore occasional detection errors
        }
      }
    }
  };

  const predictLoop = () => {
    processFrame();
    if (isPlayingRef.current) {
      requestRef.current = requestAnimationFrame(predictLoop);
    }
  };

  const analyzeWithGemini = async () => {
    if (isRecording) {
      setIsRecording(false);
      isRecordingRef.current = false;
    }

    if (recentDataRef.current.length === 0) {
      setGeminiAnalysis({ summary: "No data collected yet. Please start recording to gather data.", details: "" });
      return;
    }

    setIsAnalyzing(true);
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      let prevL: number[] | null = null;
      let prevR: number[] | null = null;

      const compactData = recentDataRef.current.map(frame => {
        const currentL = frame.hL_nose;
        const currentR = frame.hR_nose;

        const deltaL = prevL && currentL ? [currentL[0] - prevL[0], currentL[1] - prevL[1], currentL[2] - prevL[2]].map(v => parseFloat(v.toFixed(3))) : [0,0,0];
        const deltaR = prevR && currentR ? [currentR[0] - prevR[0], currentR[1] - prevR[1], currentR[2] - prevR[2]].map(v => parseFloat(v.toFixed(3))) : [0,0,0];

        prevL = currentL || prevL;
        prevR = currentR || prevR;

        return {
          t: frame.time,
          hL: currentL ? { w: currentL.map((v:number) => parseFloat(v.toFixed(3))), f: frame.leftFingerStates, d: deltaL } : null,
          hR: currentR ? { w: currentR.map((v:number) => parseFloat(v.toFixed(3))), f: frame.rightFingerStates, d: deltaR } : null,
          f: frame.face
        };
      });

      const dataString = JSON.stringify(compactData);

      const systemInstruction = `## Role
      You are a High-Speed Sign Language Interpreter (ASL/CSL specialist). You translate sequences of normalized landmark coordinates and facial blendshapes into natural English.

      ## Spatial Zone Map (Relative to Nose/Neck)
      - ZONE_FACE (y > 0.1): Origin for signs like "THINK", "KNOW", "THANK YOU".
      - ZONE_CHEST (y approx 0): Origin for signs like "ME", "MINE", "FINISH", "PLEASE".
      - ZONE_NEUTRAL: Space in front of the torso.

      ## Data Interpretation Logic
      - **Anchor Priority:** The STARTING ZONE of a movement is the strongest classifier. 
      - **Movement Physics:** - "THANK YOU" MUST start in ZONE_FACE (near lips) and move outward (+Z).
          - "FINISH" MUST start in ZONE_CHEST and move downward/outward with a hand-flip.
      - **Priority:** Hand trajectories > Spatial Anchor > Finger states > Facial Blendshapes.

      ## Output Constraints
      - Only output the requested JSON schema.
      - If a movement is ambiguous, use the Facial Blendshape (e.g., Smile vs. Pucker) as the tie-breaker.`;

      const prompt = `Context: Daily Conversation
FPS: 15
Window: ${compactData.length} Frames

## Reference Examples for Calibration:
1. CORRECT (Thank You): Start: ZONE_FACE (near mouth) -> End: ZONE_NEUTRAL. Result: "Thank you"
2. CORRECT (Finish): Start: ZONE_CHEST -> End: ZONE_NEUTRAL (sweeping down). Result: "Finished"
3. INCORRECT (Mixed): If movement starts at Chest, it CANNOT be "Thank You".

## Current Input Sequence:
${dataString}

## Task: 
Analyze the STARTING position of the hand relative to the face landmarks. Translate the sequence into natural English. Return ONLY the JSON.`;

      const response = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: prompt,
        config: {
          systemInstruction,
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              translation: {
                type: Type.STRING,
                description: "The natural English sentence translation."
              },
              reasoning: {
                type: Type.STRING,
                description: "Detailed breakdown of the hand/arm movement (trajectory, hand shape changes) and how it maps to the translation. Facial expressions should only be mentioned if they change the meaning."
              }
            },
            required: ["translation", "reasoning"]
          }
        }
      });

      try {
        const result = JSON.parse(response.text || "{}");
        setGeminiAnalysis({
          summary: result.translation || "No translation generated.",
          details: result.reasoning || ""
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

  const toggleRecording = () => {
    if (isRecording) {
      setIsRecording(false);
      isRecordingRef.current = false;
    } else {
      recentDataRef.current = [];
      setIsRecording(true);
      isRecordingRef.current = true;
      recordingStartTimeRef.current = performance.now();
      lastRecordTimeRef.current = 0;
      setGeminiAnalysis(null);
      setShowDetails(false);
    }
  };

  return (
    <div className="w-full max-w-7xl space-y-8 animate-in fade-in duration-500">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
        {/* Video Center */}
        <div className="lg:col-span-2 space-y-6">
          <div 
            className="relative w-full shadow-2xl rounded-3xl overflow-hidden bg-slate-800 border-4 border-slate-700 aspect-video"
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          >
            {isModelLoading && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/80 z-20">
                <RefreshCw className="w-10 h-10 text-indigo-500 animate-spin mb-4" />
                <p className="font-bold uppercase tracking-widest text-sm">Loading Neural Engine...</p>
              </div>
            )}
            
            {!isPlaying && mediaSource === 'none' && !isModelLoading && (
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
              autoPlay={mediaSource === 'webcam'}
              playsInline
              muted={mediaSource === 'webcam'}
              controls={false}
              onPlay={() => {
                if (mediaSource === 'video') {
                  setIsPlaying(true);
                  isPlayingRef.current = true;
                  if (requestRef.current) cancelAnimationFrame(requestRef.current);
                  predictLoop();
                }
              }}
              onPause={() => {
                if (mediaSource === 'video') {
                  setIsPlaying(false);
                  isPlayingRef.current = false;
                }
              }}
              onEnded={() => {
                if (mediaSource === 'video') {
                  setIsPlaying(false);
                  isPlayingRef.current = false;
                }
              }}
              onTimeUpdate={() => {
                if (videoRef.current) {
                  setVideoProgress(videoRef.current.currentTime);
                }
              }}
              onLoadedMetadata={() => {
                if (videoRef.current) {
                  setVideoDuration(videoRef.current.duration);
                }
              }}
              onSeeked={() => {
                if (mediaSource === 'video' && !isPlayingRef.current) {
                  processFrame();
                }
              }}
              className={`absolute inset-0 w-full h-full object-contain ${mediaSource === 'webcam' ? '-scale-x-100' : ''}`}
            />
            <canvas
              ref={canvasRef}
              className={`absolute inset-0 w-full h-full object-contain pointer-events-none ${mediaSource === 'webcam' ? '-scale-x-100' : ''}`}
              style={{ imageRendering: 'auto' }}
            />

            {/* Floating Emojis */}
            <div className="pointer-events-none absolute inset-0 z-20 overflow-hidden">
              {popups.map(p => (
                <div
                  key={p.id}
                  className="absolute bottom-0 floating-emoji"
                  style={{
                    left: `${p.left}%`,
                    fontSize: `${p.size}rem`,
                    '--duration': `${p.duration}s`,
                    '--tx': p.tx,
                  } as React.CSSProperties}
                >
                  {p.emoji}
                </div>
              ))}
            </div>

            {/* Custom Video Controls */}
            {mediaSource === 'video' && (
              <div 
                className={`absolute bottom-0 left-0 right-0 bg-slate-900/90 backdrop-blur-md p-4 flex items-center gap-4 z-30 border-t-4 border-black shadow-[0_-4px_0_0_rgba(0,0,0,1)] transition-all duration-300 ease-in-out ${
                  isHoveringVideo || !isPlaying ? 'translate-y-0 opacity-100' : 'translate-y-8 opacity-0 pointer-events-none'
                }`}
              >
                <button
                  onClick={() => {
                    if (videoRef.current) {
                      if (isPlaying) {
                        videoRef.current.pause();
                      } else {
                        videoRef.current.play();
                      }
                    }
                  }}
                  className="text-white hover:text-indigo-400 transition-colors active:scale-95 focus:outline-none focus:ring-2 focus:ring-indigo-500 rounded-full p-1"
                  title={isPlaying ? "Pause" : "Play"}
                >
                  {isPlaying ? <Pause className="w-6 h-6 fill-current" /> : <Play className="w-6 h-6 fill-current" />}
                </button>
                <button
                  onClick={() => {
                    if (videoRef.current) {
                      videoRef.current.currentTime = 0;
                      videoRef.current.play();
                    }
                  }}
                  className="text-white hover:text-emerald-400 transition-colors active:scale-95 focus:outline-none focus:ring-2 focus:ring-emerald-500 rounded-full p-1"
                  title="Restart"
                >
                  <RefreshCw className="w-6 h-6" />
                </button>
                <div className="flex-1 flex items-center gap-3">
                  <span className="text-xs font-mono font-black text-slate-300 w-10 text-right">
                    {Math.floor(videoProgress / 60)}:{(Math.floor(videoProgress % 60)).toString().padStart(2, '0')}
                  </span>
                  <input
                    type="range"
                    min="0"
                    max={videoDuration || 100}
                    step="0.01"
                    value={videoProgress}
                    onChange={(e) => {
                      if (videoRef.current) {
                        videoRef.current.currentTime = Number(e.target.value);
                      }
                    }}
                    className="flex-1 h-3 bg-slate-700 rounded-full appearance-none cursor-pointer accent-indigo-500 border-2 border-black focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                  <span className="text-xs font-mono font-black text-slate-300 w-10">
                    {Math.floor(videoDuration / 60)}:{(Math.floor(videoDuration % 60)).toString().padStart(2, '0')}
                  </span>
                </div>
              </div>
            )}
          </div>

          <div className="flex flex-col sm:flex-row justify-center gap-4 w-full">
            <button
              onClick={toggleCamera}
              disabled={isModelLoading}
              className={`flex items-center justify-center w-full sm:w-auto border-4 border-black px-6 py-3 sm:px-8 sm:py-4 rounded-2xl font-black text-white shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] transition-all active:translate-y-1 active:shadow-none uppercase text-xs sm:text-sm whitespace-nowrap focus:outline-none focus:ring-4 focus:ring-indigo-500 ${
                mediaSource === 'webcam' ? 'bg-rose-600 hover:bg-rose-500' : 'bg-indigo-600 hover:bg-indigo-500'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {mediaSource === 'webcam' ? 'Disable Camera' : 'Enable Camera'}
            </button>
            
            <input 
              type="file" 
              accept="video/*" 
              className="hidden" 
              ref={fileInputRef} 
              onChange={handleFileUpload} 
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isModelLoading}
              className="flex items-center justify-center w-full sm:w-auto bg-violet-600 hover:bg-violet-500 border-4 border-black px-6 py-3 sm:px-8 sm:py-4 rounded-2xl font-black text-white shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] transition-all active:translate-y-1 active:shadow-none uppercase text-xs sm:text-sm whitespace-nowrap disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-4 focus:ring-violet-500"
            >
              Upload Video
            </button>

            <button
              onClick={() => {
                const newVal = !showLandmarks;
                setShowLandmarks(newVal);
                showLandmarksRef.current = newVal;
              }}
              disabled={mediaSource === 'none'}
              className="flex items-center justify-center w-full sm:w-auto bg-slate-700 hover:bg-slate-600 border-4 border-black px-6 py-3 sm:px-8 sm:py-4 rounded-2xl font-black text-white shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] transition-all active:translate-y-1 active:shadow-none uppercase text-xs sm:text-sm whitespace-nowrap disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-4 focus:ring-slate-400"
            >
              {showLandmarks ? 'Hide Skeleton' : 'Show Skeleton'}
            </button>
          </div>
        </div>

        {/* Expanded Metrics Sidebar */}
        <div className="space-y-6 flex flex-col">
          <div className="bg-slate-800/80 border-4 border-black p-4 rounded-[2.5rem] backdrop-blur-md w-full shadow-[10px_10px_0px_0px_rgba(0,0,0,1)] text-white">
            <h2 className="text-xl font-black mb-4 border-b-4 border-black pb-2 flex justify-between items-center uppercase italic">
              Sign Analytics
              {mediaSource === 'webcam' && (
                <span className="text-[10px] bg-red-500 text-white px-3 py-1 rounded-full border-2 border-black animate-pulse">
                  LIVE
                </span>
              )}
            </h2>

            <div className="space-y-3">
              {/* Group 1: Hands */}
              <div className="space-y-1 border-b border-slate-700/50 pb-2">
                <MetricBar label="Hand Activity" value={handActivity} color="bg-emerald-400" textColor="text-emerald-400" />
              </div>

              {/* Group 2: Face */}
              <div className="space-y-1 border-b border-slate-700/50 pb-2">
                <MetricBar label="Facial Expressiveness" value={faceActivity} color="bg-blue-400" textColor="text-blue-400" />
              </div>
            </div>
          </div>

          {/* Emoji Matching Section */}
          <div className="bg-slate-800/80 border-4 border-black p-4 rounded-[2.5rem] backdrop-blur-md w-full shadow-[10px_10px_0px_0px_rgba(0,0,0,1)] text-white flex-1 flex flex-col">
            <div className="flex justify-between items-center mb-4 border-b-4 border-black pb-2">
              <h2 className="text-xl font-black uppercase italic">
                Emoji Matcher
              </h2>
              <label className="flex items-center gap-2 cursor-pointer bg-slate-700/50 px-3 py-1.5 rounded-xl border-2 border-slate-600 hover:bg-slate-700 transition-colors">
                <input 
                  type="checkbox" 
                  checked={interactiveAnimation}
                  onChange={(e) => setInteractiveAnimation(e.target.checked)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      setInteractiveAnimation(!interactiveAnimation);
                    }
                  }}
                  className="w-4 h-4 accent-emerald-500 rounded cursor-pointer focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                />
                <span className="text-[10px] font-bold uppercase tracking-wider">Animation</span>
              </label>
            </div>
            <div className="grid grid-cols-3 gap-1 sm:gap-2 flex-1">
              {[
                { emoji: '✌️', label: 'Yeah' },
                { emoji: '👍', label: 'Thumbs Up' },
                { emoji: '👌', label: 'OK' },
                { emoji: '🖐️', label: 'Open' },
                { emoji: '✊', label: 'Fist' },
                { emoji: '🫶', label: 'Heart' },
                { emoji: '🤫', label: 'Silence' },
                { emoji: '😕', label: 'Confused' },
                { emoji: '🤔', label: 'Think' },
              ].map((item) => {
                const canAnimate = ['👌', '🫶', '🤫', '✌️', '👍'].includes(item.emoji);
                const isBright = interactiveAnimation && canAnimate;
                return (
                <div 
                  key={item.emoji}
                  className={`flex flex-col items-center justify-center p-1 sm:p-2 rounded-xl sm:rounded-2xl border-2 sm:border-4 transition-all duration-300 ${
                    detectedEmoji === item.emoji 
                      ? 'bg-emerald-500 border-black shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] sm:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] scale-105' 
                      : isBright ? 'bg-slate-600 border-emerald-500/30 opacity-100 shadow-[0_0_10px_rgba(16,185,129,0.2)] sm:shadow-[0_0_15px_rgba(16,185,129,0.2)]' : 'bg-slate-700 border-transparent opacity-50'
                  }`}
                >
                  <span className="text-xl sm:text-3xl mb-0.5 sm:mb-1">{item.emoji}</span>
                  <span className="text-[6px] sm:text-[9px] font-bold uppercase tracking-wider text-center leading-tight">{item.label}</span>
                </div>
              )})}
            </div>
          </div>
        </div>
      </div>

      {/* Activity Timeline */}
      <div className="bg-slate-800/80 border-4 border-black p-4 rounded-[2.5rem] backdrop-blur-md w-full shadow-[10px_10px_0px_0px_rgba(0,0,0,1)] text-white mb-8">
        <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-3 mb-4 border-b-4 border-black pb-3">
          <h2 className="text-xl font-black uppercase italic flex flex-wrap items-center gap-x-2">
            <span>Activity</span>
            <span className="inline-flex items-center gap-2">
              Timeline
            </span>
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
            <LineChart data={activityHistory} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
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
                tickFormatter={(val) => val === 0 ? 'Low' : val === 1 ? 'High' : val.toFixed(2)}
              />
              <Tooltip 
                labelFormatter={(label) => new Date(label).toLocaleTimeString()}
                formatter={(value: number) => [value.toFixed(2), 'Activity Score']}
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
      <div className="hidden bg-indigo-950/80 border-4 border-indigo-500 p-4 rounded-[2.5rem] backdrop-blur-md w-full shadow-[10px_10px_0px_0px_rgba(99,102,241,0.4)] text-white">
        <h2 className="text-lg font-black mb-4 border-b-4 border-indigo-500/50 pb-2 flex justify-between items-center uppercase italic text-indigo-300">
          AI Sign Language Translator
          <Hand className="w-5 h-5 text-indigo-400" />
        </h2>

        <div className="flex flex-col lg:flex-row gap-6">
          {/* Left Side: Controls */}
          <div className="w-full lg:w-1/3 space-y-3">
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
                  <option value={33}>30 fps</option>
                  <option value={50}>20 fps</option>
                  <option value={66}>15 fps</option>
                  <option value={100}>10 fps</option>
                </select>

                <button
                  onClick={toggleRecording}
                  disabled={mediaSource === 'none' || isAnalyzing}
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
              disabled={isAnalyzing || (!isRecording && recentDataRef.current.length === 0)}
              className="w-full bg-indigo-500 hover:bg-indigo-400 border-4 border-black px-4 py-3 rounded-2xl font-black text-white shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] transition-all active:translate-y-1 active:shadow-none uppercase text-xs flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-4 focus:ring-indigo-500"
            >
              {isAnalyzing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Translating...
                </>
              ) : (
                <>
                  <Hand className="w-4 h-4" />
                  Translate Sign
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
                  Translation
                </div>
                <p className="font-medium text-xl text-white leading-snug">{geminiAnalysis.summary}</p>

                {geminiAnalysis.details && (
                  <div className="mt-auto pt-2">
                    <button
                      onClick={() => setShowDetails(!showDetails)}
                      className="flex items-center gap-1 text-[10px] font-black uppercase tracking-widest text-indigo-400 hover:text-indigo-300 transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 rounded px-1"
                    >
                      {showDetails ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                      {showDetails ? "Hide Reasoning" : "Show Reasoning"}
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
                  Awaiting data collection. Start recording to capture sign language metrics, then request an AI translation.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes floatUp {
          0% {
            transform: translateY(100px) translateX(0) rotate(0deg) scale(0.5);
            opacity: 0;
          }
          10% {
            opacity: 1;
            transform: translateY(0px) translateX(calc(var(--tx) * 0.2px)) rotate(10deg) scale(1);
          }
          90% {
            opacity: 1;
          }
          100% {
            transform: translateY(-80vh) translateX(calc(var(--tx) * 1px)) rotate(calc(var(--tx) * 1deg)) scale(1.5);
            opacity: 0;
          }
        }
        .floating-emoji {
          animation: floatUp var(--duration) ease-out forwards;
        }
      `}</style>
    </div>
  );
}

function MetricBar({ label, value, color, textColor, height = "h-3" }: { label: string, value: number, color: string, textColor: string, height?: string }) {
  const percent = Math.min(Math.max(Math.round(value * 100), 0), 100);
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
