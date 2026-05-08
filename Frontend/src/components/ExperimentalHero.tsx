import { useState, useRef, useEffect } from "react";
import { useTranslation, Trans } from "react-i18next";
import { useTheme } from "../context/ThemeContext";
import { clsx } from "clsx";
import AudioInput from "./AudioInput/AudioInput"; // Using the new component
import AudioPlayer from "./AudioPlayer";
import Result from "./Result";
import axios from "axios";
import { convertFileToWav } from "../utils/audioUtils";
import { AnalysisResult } from "../types";


type JointResult = {
  name: string;
  model: string;
  url: string;
  color: string;
  emotion: string;
  confidence: number;
  status: 'loading' | 'done' | 'error';
  error?: string;
};

const JOINT_MODELS = [
  { name: 'Gelişmiş Cümle Analizi', url: '/api/predict_advanced_sentence', color: '#ef4444' },
  { name: 'HuBERT (HuggingFace)', url: '/analyze_hubert', color: '#10b981' },
  { name: 'Wav2Vec2 Turkish', url: '/analyze_wav2vec2_turkish', color: '#0ea5e9' },
  { name: 'SUPERB Wav2Vec2-ER', url: '/analyze_superb', color: '#14b8a6' },
  { name: 'SenseVoice (Alibaba)', url: '/analyze_sensevoice', color: '#f97316' },
  { name: 'ExHuBERT', url: '/analyze_exhubert', color: '#8b5cf6' },
  { name: 'WavLM Base Plus', url: '/analyze_wavlm_base_plus', color: '#6366f1' },
  { name: 'WavLM Large', url: '/analyze_wavlm_large', color: '#f43f5e' },
];

const parseConf = (c: any): number => {
  if (typeof c === 'string') return parseFloat(c.replace('%', ''));
  if (typeof c === 'number') return c <= 1 ? c * 100 : c;
  return 0;
};

// Helper to generate levels for visualization from a file
const generateWaveLevels = async (file: File, bars: number): Promise<number[]> => {
  const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
  const ctx = new AudioCtx();
  const ab = await file.arrayBuffer();
  const audioBuffer = await ctx.decodeAudioData(ab);
  const rawData = audioBuffer.getChannelData(0);
  const blockSize = Math.floor(rawData.length / bars);
  const samples = [];
  for (let i = 0; i < bars; i++) {
    let sum = 0;
    for (let j = 0; j < blockSize; j++) {
      sum += Math.abs(rawData[i * blockSize + j]);
    }
    samples.push(sum / blockSize);
  }
  const max = Math.max(...samples, 0.01);
  return samples.map(s => s / max);
};

const Hero = () => {
  const { t, i18n } = useTranslation();
  const { isDark } = useTheme();
  const isTr = i18n.language === 'tr';

  // Base models list
  const BASE_MODELS = [
    { id: 'catboost', name: 'CatBoost' },
    { id: 'xgboost', name: 'XGBoost' },
    { id: 'lightgbm', name: 'LightGBM' },
    { id: 'rf', name: 'Random Forest' },
    { id: 'knn', name: 'K-Nearest Neighbors' },
    { id: 'svm', name: 'Support Vector Machine' },
    { id: 'mlp', name: 'Multi-Layer Perceptron' },
    { id: 'gradient_boosting', name: 'Gradient Boosting' },
    { id: 'dnn', name: 'Deep Neural Network (DNN)' },
    { id: 'cnn1d', name: '1D Convolutional Neural Network' },
  ];

  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [savedLevels, setSavedLevels] = useState<number[]>([]);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [selectedModel, setSelectedModel] = useState('catboost');
  const [qualityMode, setQualityMode] = useState<'studio' | 'robust'>('robust');
  const [mode, setMode] = useState<'word' | 'sentence_segmented' | 'sentence_whole' | 'advanced_sentence' | 'hubert' | 'wav2vec2_turkish' | 'superb_er' | 'sensevoice' | 'exhubert' | 'wavlm_base_plus' | 'wavlm_large' | 'joint_test'>('word');
  const [sttEngine, setSttEngine] = useState<'vad' | 'vosk' | 'whisperx'>('vad');
  const [jointResults, setJointResults] = useState<JointResult[]>([]);
  const [isJointTesting, setIsJointTesting] = useState(false);
  const [jointDone, setJointDone] = useState(false);

  // Compute actual backend key
  const activeModelKey = selectedModel === 'majority_voting'
    ? 'majority_voting'
    : (qualityMode === 'robust' ? `${selectedModel}_robust` : selectedModel);
  const activeModelName = selectedModel === 'majority_voting'
    ? t('majority_voting')
    : `${BASE_MODELS.find(m => m.id === selectedModel)?.name} (${qualityMode === 'robust' ? t('noisy') : t('studio')})`;

  // Segmentation and Next flow - stores results for ALL engines
  type SegmentItem = { start: number, end: number, word?: string, emotion?: string };
  type AllSegResults = {
    vad?: { segments: SegmentItem[], error?: string, elapsed?: number },
    vosk?: { segments: SegmentItem[], error?: string, elapsed?: number },
    whisperx?: { segments: SegmentItem[], error?: string, elapsed?: number },
  };
  const [allSegmentResults, setAllSegmentResults] = useState<AllSegResults>({});
  const [activeSegTab, setActiveSegTab] = useState<'vad' | 'vosk' | 'whisperx'>('vad');
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [showModelSelection, setShowModelSelection] = useState(false);

  // Computed: currently visible segments based on active tab
  const sentenceSegments = allSegmentResults[activeSegTab]?.segments || null;
  const segError = allSegmentResults[activeSegTab]?.error || null;
  const hasAnyResults = Object.keys(allSegmentResults).length > 0;

  // Audio Player State
  const [isPlaying, setIsPlaying] = useState(false);
  const [playProgress, setPlayProgress] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [isSpeedMenuOpen, setIsSpeedMenuOpen] = useState(false);

  const audioElementRef = useRef<HTMLAudioElement>(null);
  const playbackAnimationRef = useRef<number | null>(null);

  const handleAudioReady = async (file: File) => {
    setAudioFile(file);
    const url = URL.createObjectURL(file);
    setRecordedUrl(url);

    // Generate levels for preview
    const levels = await generateWaveLevels(file, 60);
    setSavedLevels(levels);

    if (mode === 'sentence_segmented' || mode === 'advanced_sentence' || mode === 'hubert' || mode === 'wav2vec2_turkish' || mode === 'superb_er' || mode === 'sensevoice' || mode === 'exhubert' || mode === 'wavlm_base_plus' || mode === 'wavlm_large' || mode === 'joint_test') {
      setShowModelSelection(false);
    } else {
      setShowModelSelection(true);
    }
  };

  const handleSegmentation = async (file: File) => {
    setIsSegmenting(true);
    setAllSegmentResults({});
    try {
      const wavBlob = await convertFileToWav(file);

      // 3 motoru paralel çalıştır
      const [vadResult, voskResult, whisperxResult] = await Promise.allSettled([
        // 1. VAD (eski yöntem)
        (async () => {
          const fd = new FormData();
          fd.append("file", wavBlob, "converted_audio.wav");
          fd.append("model_type", activeModelKey);
          const resp = await axios.post(`${import.meta.env.VITE_API_URL}/segment-sentence`, fd, {
            headers: { 'Content-Type': 'multipart/form-data' }
          });
          return { segments: resp.data.segments as SegmentItem[], elapsed: resp.data.elapsed_seconds as number };
        })(),
        // 2. Vosk
        (async () => {
          const fd = new FormData();
          fd.append("file", wavBlob, "converted_audio.wav");
          fd.append("stt_engine", "vosk");
          fd.append("model_type", activeModelKey);
          const resp = await axios.post(`${import.meta.env.VITE_API_URL}/transcribe`, fd, {
            headers: { 'Content-Type': 'multipart/form-data' }
          });
          return { segments: (resp.data.words || []).map((w: any) => ({ start: w.start, end: w.end, word: w.word, emotion: w.emotion })) as SegmentItem[], elapsed: resp.data.elapsed_seconds as number };
        })(),
        // 3. WhisperX
        (async () => {
          const fd = new FormData();
          fd.append("file", wavBlob, "converted_audio.wav");
          fd.append("stt_engine", "whisperx");
          fd.append("model_type", activeModelKey);
          const resp = await axios.post(`${import.meta.env.VITE_API_URL}/transcribe`, fd, {
            headers: { 'Content-Type': 'multipart/form-data' }
          });
          return { segments: (resp.data.words || []).map((w: any) => ({ start: w.start, end: w.end, word: w.word, emotion: w.emotion })) as SegmentItem[], elapsed: resp.data.elapsed_seconds as number };
        })(),
      ]);

      const results: AllSegResults = {};
      if (vadResult.status === 'fulfilled') {
        results.vad = { segments: vadResult.value.segments, elapsed: vadResult.value.elapsed };
      } else {
        results.vad = { segments: [], error: String(vadResult.reason?.response?.data?.error || vadResult.reason?.message || 'Hata') };
      }
      if (voskResult.status === 'fulfilled') {
        results.vosk = { segments: voskResult.value.segments, elapsed: voskResult.value.elapsed };
      } else {
        results.vosk = { segments: [], error: String(voskResult.reason?.response?.data?.error || voskResult.reason?.message || 'Hata') };
      }
      if (whisperxResult.status === 'fulfilled') {
        results.whisperx = { segments: whisperxResult.value.segments, elapsed: whisperxResult.value.elapsed };
      } else {
        results.whisperx = { segments: [], error: String(whisperxResult.reason?.response?.data?.error || whisperxResult.reason?.message || 'Hata') };
      }

      setAllSegmentResults(results);
      // İlk başarılı sonucu aktif tab olarak seç
      if (results.vad && !results.vad.error) setActiveSegTab('vad');
      else if (results.vosk && !results.vosk.error) setActiveSegTab('vosk');
      else if (results.whisperx && !results.whisperx.error) setActiveSegTab('whisperx');

    } catch (err) {
      console.error(err);
      alert("Cümle ayrıştırma hatası.");
    } finally {
      setIsSegmenting(false);
    }
  }

  const tooglePlayPause = () => {
    if (!audioElementRef.current) return;
    if (isPlaying) {
      audioElementRef.current.pause();
      setIsPlaying(false);
    } else {
      audioElementRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleSpeedChange = (rate: number) => {
    setPlaybackRate(rate);
    if (audioElementRef.current) audioElementRef.current.playbackRate = rate;
  };

  const updatePlayProgress = () => {
    if (audioElementRef.current) {
      setPlayProgress(audioElementRef.current.currentTime / audioElementRef.current.duration);
      setCurrentTime(audioElementRef.current.currentTime);
      playbackAnimationRef.current = requestAnimationFrame(updatePlayProgress);
    }
  };

  useEffect(() => {
    if (isPlaying) {
      playbackAnimationRef.current = requestAnimationFrame(updatePlayProgress);
    } else {
      if (playbackAnimationRef.current) cancelAnimationFrame(playbackAnimationRef.current);
    }
    return () => {
      if (playbackAnimationRef.current) cancelAnimationFrame(playbackAnimationRef.current);
    };
  }, [isPlaying]);

  const handleAnalyze = async () => {
    if (!audioFile) return;
    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      const wavBlob = await convertFileToWav(audioFile);
      const formData = new FormData();
      formData.append("file", wavBlob, "converted_audio.wav");
      formData.append("model_type", activeModelKey);
      formData.append("quality", qualityMode);
      if (sttEngine !== 'vad') {
        formData.append("stt_engine", sttEngine);
      }

      // Determine endpoint
      let endpoint: string;
      if (mode === 'hubert') {
        endpoint = '/analyze_hubert';
      } else if (mode === 'exhubert') {
        endpoint = '/analyze_exhubert';
      } else if (mode === 'wav2vec2_turkish') {
        endpoint = '/analyze_wav2vec2_turkish';
      } else if (mode === 'superb_er') {
        endpoint = '/analyze_superb';
      } else if (mode === 'sensevoice') {
        endpoint = '/analyze_sensevoice';
      } else if (mode === 'wavlm_base_plus') {
        endpoint = '/analyze_wavlm_base_plus';
      } else if (mode === 'wavlm_large') {
        endpoint = '/analyze_wavlm_large';
      } else if (selectedModel === 'majority_voting') {
        endpoint = '/analyze_voting';
      } else if (mode === 'word') {
        endpoint = '/predict';
      } else if (mode === 'sentence_segmented') {
        endpoint = '/predict-sentence';
      } else if (mode === 'advanced_sentence') {
        endpoint = '/api/predict_advanced_sentence';
      } else {
        endpoint = '/api/predict_sentence_whole';
      }

      const response = await axios.post(`${import.meta.env.VITE_API_URL}${endpoint}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      // Parse response based on endpoint type
      let emotion, confidence, all_scores;

      if (mode === 'hubert' || mode === 'wav2vec2_turkish' || mode === 'superb_er' || mode === 'sensevoice' || mode === 'exhubert' || mode === 'wavlm_base_plus' || mode === 'wavlm_large') {
        emotion = response.data.emotion;
        confidence = response.data.confidence;
        all_scores = response.data.all_scores;
      } else if (selectedModel === 'majority_voting') {
        // Voting endpoint returns standard format: { emotion, confidence, all_scores, model_details }
        emotion = response.data.emotion;
        confidence = response.data.confidence;
        all_scores = response.data.all_scores;
      } else if (mode === 'sentence_segmented') {
        // Segmented Sentence endpoint returns final_emotion
        emotion = response.data.final_emotion;
        confidence = response.data.confidence;
        if (response.data.weighted_details) {
          all_scores = response.data.weighted_details;
        }
      } else if (mode === 'sentence_whole') {
        // Whole Sentence endpoint returns 'emotion' directly
        emotion = response.data.emotion;
        confidence = response.data.confidence;
        all_scores = response.data.all_scores;
      } else if (mode === 'advanced_sentence') {
        emotion = response.data.emotion;
        confidence = response.data.confidence;
        all_scores = response.data.all_scores;
      } else {
        // Standard /predict response
        emotion = response.data.emotion;
        confidence = response.data.confidence;
        all_scores = response.data.all_scores;
      }

      // Parse confidence string (e.g., "%95.20" -> 95.20)
      let confidenceValue = 0;
      if (typeof confidence === 'string') {
        confidenceValue = parseFloat(confidence.replace('%', ''));
      } else if (typeof confidence === 'number') {
        confidenceValue = confidence;
      }

      // Backend returns 0-100 range, Frontend expects 0-1 range
      const normalizedConfidence = confidenceValue / 100;

      // Process all_scores if available (Backend returns 0-100)
      const emotionsMap: { [key: string]: number; happy: number; sad: number; angry: number; calm: number; } = {
        happy: 0,
        sad: 0,
        angry: 0,
        calm: 0
      };
      if (all_scores) {
        Object.entries(all_scores).forEach(([key, val]) => {
          emotionsMap[key.toLowerCase()] = (val as number) / 100;
        });
      } else {
        // Fallback for older backend response
        const fallbackEmotion = (emotion || 'neutral').toLowerCase();
        emotionsMap[fallbackEmotion] = normalizedConfidence;
      }

      // Normalize total so bars aren't broken if sum exceeds 100% due to weights
      const totalSum = Object.values(emotionsMap).reduce((a, b) => a + b, 0);
      if (totalSum > 0) {
        Object.keys(emotionsMap).forEach(key => {
          emotionsMap[key] = emotionsMap[key] / totalSum;
        });
      }

      // Handle word_timestamps
      let finalWordTimestamps = response.data.word_timestamps || [];

      // Eğer sentence modundaysak ve halihazırda ayrıştırılmış segmentler (ve duyguları) varsa onları kullan
      if ((mode === 'sentence_segmented' || mode === 'sentence_whole') && sentenceSegments && sentenceSegments.length > 0) {
        finalWordTimestamps = sentenceSegments.map((seg, i) => ({
          word: seg.word || `${t('word_label')} ${i + 1}`,
          start: seg.start,
          end: seg.end,
          emotion: seg.emotion || 'neutral',
          confidence: 1.0 // Mock confidence since client-side segments don't have it
        }));
      }

      setAnalysisResult({
        dominant_emotion: emotion,
        emotions: emotionsMap,
        confidence: normalizedConfidence,
        word_timestamps: finalWordTimestamps,
        model_details: response.data.model_details || undefined,
        frequency_data: response.data.frequency_data || undefined
      });

      setIsAnalyzing(false);

    } catch (error: any) {
      console.error("Analysis Request Failed:", error);
      setIsAnalyzing(false);
      let msg = t('analysis_failed') || "Analiz hatası.";
      if (error.response?.data?.error) {
        msg = `Hata: ${error.response.data.error}`;
      } else if (error.message) {
        msg = `Hata: ${error.message}`;
      }
      alert(msg);
    }
  };

  const handleJointTest = async () => {
    if (!audioFile) return;
    setIsJointTesting(true);
    setJointDone(false);
    setJointResults(JOINT_MODELS.map(m => ({ ...m, model: m.name, emotion: '', confidence: 0, status: 'loading' as const })));

    try {
      const wavBlob = await convertFileToWav(audioFile);
      await Promise.allSettled(
        JOINT_MODELS.map(async (m, i) => {
          try {
            const fd = new FormData();
            fd.append('file', wavBlob, 'audio.wav');
            const resp = await axios.post(`${import.meta.env.VITE_API_URL}${m.url}`, fd, {
              headers: { 'Content-Type': 'multipart/form-data' }
            });
            const emotion = resp.data.emotion || resp.data.final_emotion || 'unknown';
            const confidence = parseConf(resp.data.confidence);
            setJointResults(prev => {
              const next = [...prev];
              next[i] = { ...next[i], emotion, confidence, status: 'done' };
              return next;
            });
          } catch (err: any) {
            setJointResults(prev => {
              const next = [...prev];
              next[i] = { ...next[i], emotion: 'error', confidence: 0, status: 'error', error: err?.response?.data?.error || err?.message || 'Hata' };
              return next;
            });
          }
        })
      );
    } catch (err) {
      console.error('Joint test error:', err);
    }

    setIsJointTesting(false);
    setJointDone(true);
  };

  const reset = () => {
    setAudioFile(null);
    setRecordedUrl(null);
    setAnalysisResult(null);
    setSavedLevels([]);
    setIsPlaying(false);
    setPlayProgress(0);
    setAllSegmentResults({});
    setActiveSegTab('vad');
    setShowModelSelection(false);
    setSttEngine('vad');
    setJointResults([]);
    setIsJointTesting(false);
    setJointDone(false);
  };

  return (
    <div
      className={clsx(
        "relative w-full flex-grow flex flex-col items-center font-sans",
        (analysisResult || jointDone || hasAnyResults || !!audioFile) ? "justify-start overflow-x-hidden" : "justify-start"
      )}
      style={(analysisResult || jointDone || hasAnyResults || !!audioFile) ? { paddingTop: '80px', paddingBottom: '24px' } : { paddingTop: '120px', paddingBottom: '32px' }}
    >



      <div className={`relative z-10 w-full max-w-6xl px-4 md:px-6 flex flex-col items-center${!(analysisResult || jointDone || hasAnyResults || !!audioFile) ? ' my-auto' : ''}`} style={(analysisResult || jointDone) ? { paddingBottom: '4px' } : { paddingBottom: '40px' }}>


        <div className={`
          relative w-full shadow-2xl transition-all duration-1000 ease-[cubic-bezier(0.16,1,0.3,1)]
          flex flex-col items-center justify-center border
          ${isDark ? "bg-[#0f172a]/70 backdrop-blur-[40px] border-white/10 shadow-[0_0_100px_rgba(99,102,241,0.15)]" : "bg-white/60 backdrop-blur-[40px] border-indigo-100 shadow-[0_0_60px_rgba(99,102,241,0.08)]"}
          ${(analysisResult || jointDone) ? "max-w-[100vw] sm:max-w-[98vw] lg:max-w-[1600px] overflow-visible rounded-3xl md:rounded-[2.5rem] mx-auto border-indigo-500/20" : "max-w-5xl min-h-[320px] px-4 py-8 sm:p-6 md:p-10 rounded-[2rem] md:rounded-[3rem] w-full mx-4 md:mx-0"}
        `}
          style={!(analysisResult || jointDone) ? { minHeight: '380px' } : { marginTop: '0', padding: '0 8px 4px' }}
        >

          {/* Subtle Inner Glow */}
          {!(analysisResult || jointDone) && (
            <div className="absolute inset-0 rounded-[3rem] pointer-events-none shadow-[inset_0_0_60px_rgba(255,255,255,0.05)]" />
          )}
          {/* Mode Switcher */}
          {!(analysisResult || jointDone) && (
            <div className="w-full relative mt-6">
              <div className="mode-btn-wrap">
                <button
                  onClick={() => { setMode('word'); setShowModelSelection(true); }}
                  className={`mode-btn ${mode === 'word'
                    ? 'bg-blue-600 text-white shadow-sm border-blue-500'
                    : isDark ? 'bg-slate-700 text-blue-400 hover:text-blue-300 border-blue-500' : 'bg-white text-blue-600 hover:text-blue-700 border-blue-400'
                    }`}
                >
                  {t('mode_word')}
                </button>
                <button
                  onClick={() => { setMode('sentence_segmented'); setShowModelSelection(false); setAllSegmentResults({}); }}
                  className={`mode-btn ${mode === 'sentence_segmented'
                    ? 'bg-fuchsia-600 text-white shadow-sm border-fuchsia-500'
                    : isDark ? 'bg-slate-700 text-fuchsia-400 hover:text-fuchsia-300 border-fuchsia-500' : 'bg-white text-fuchsia-600 hover:text-fuchsia-700 border-fuchsia-400'
                    }`}
                >
                  {t('mode_sentence_segmented')}
                </button>
                <button
                  onClick={() => { setMode('sentence_whole'); setShowModelSelection(true); setAllSegmentResults({}); }}
                  className={`mode-btn ${mode === 'sentence_whole'
                    ? 'bg-cyan-600 text-white shadow-sm border-cyan-500'
                    : isDark ? 'bg-slate-700 text-cyan-400 hover:text-cyan-300 border-cyan-500' : 'bg-white text-cyan-600 hover:text-cyan-700 border-cyan-400'
                    }`}
                >
                  {t('mode_sentence_whole')}
                </button>
                <button
                  onClick={() => { setMode('advanced_sentence'); setShowModelSelection(false); setAllSegmentResults({}); }}
                  className={`mode-btn ${mode === 'advanced_sentence'
                    ? 'bg-red-600 text-white shadow-sm border-red-500'
                    : isDark ? 'bg-slate-700 text-red-400 hover:text-red-300 border-red-500' : 'bg-white text-red-500 hover:text-red-600 border-red-400'
                    }`}
                >
                  {isTr ? "Gelişmiş Cümle Analizi" : "Advanced Sentence Analysis"}
                </button>
                <button
                  onClick={() => { setMode('hubert'); setShowModelSelection(false); setAllSegmentResults({}); }}
                  className={`mode-btn ${mode === 'hubert'
                    ? 'bg-emerald-600 text-white shadow-sm border-emerald-500'
                    : isDark ? 'bg-slate-700 text-emerald-400 hover:text-emerald-300 border-emerald-500' : 'bg-white text-emerald-600 hover:text-emerald-700 border-emerald-400'
                    }`}
                >
                  HuBERT (HuggingFace)
                </button>
                <button
                  onClick={() => { setMode('wav2vec2_turkish'); setShowModelSelection(false); setAllSegmentResults({}); }}
                  className={`mode-btn ${mode === 'wav2vec2_turkish'
                    ? 'bg-sky-600 text-white shadow-sm border-sky-500'
                    : isDark ? 'bg-slate-700 text-sky-400 hover:text-sky-300 border-sky-500' : 'bg-white text-sky-600 hover:text-sky-700 border-sky-400'
                    }`}
                >
                  Wav2Vec2 Turkish
                </button>
                <button
                  onClick={() => { setMode('superb_er'); setShowModelSelection(false); setAllSegmentResults({}); }}
                  className={`mode-btn ${mode === 'superb_er'
                    ? 'bg-teal-600 text-white shadow-sm border-teal-500'
                    : isDark ? 'bg-slate-700 text-teal-400 hover:text-teal-300 border-teal-500' : 'bg-white text-teal-600 hover:text-teal-700 border-teal-400'
                    }`}
                >
                  SUPERB Wav2Vec2-ER
                </button>
                <button
                  onClick={() => { setMode('sensevoice'); setShowModelSelection(false); setAllSegmentResults({}); }}
                  className={`mode-btn ${mode === 'sensevoice'
                    ? 'bg-orange-600 text-white shadow-sm border-orange-500'
                    : isDark ? 'bg-slate-700 text-orange-400 hover:text-orange-300 border-orange-500' : 'bg-white text-orange-600 hover:text-orange-700 border-orange-400'
                    }`}
                >
                  SenseVoice (Alibaba)
                </button>
                <button
                  onClick={() => { setMode('exhubert'); setShowModelSelection(false); setAllSegmentResults({}); }}
                  className={`mode-btn ${mode === 'exhubert'
                    ? 'bg-violet-600 text-white shadow-sm border-violet-500'
                    : isDark ? 'bg-slate-700 text-violet-400 hover:text-violet-300 border-violet-500' : 'bg-white text-violet-600 hover:text-violet-700 border-violet-400'
                    }`}
                >
                  ExHuBERT
                </button>
                <button
                  onClick={() => { setMode('wavlm_base_plus'); setShowModelSelection(false); setAllSegmentResults({}); }}
                  className={`mode-btn ${mode === 'wavlm_base_plus'
                    ? 'bg-indigo-800 text-white shadow-sm border-indigo-700'
                    : isDark ? 'bg-slate-700 text-indigo-300 hover:text-indigo-200 border-indigo-600' : 'bg-white text-indigo-800 hover:text-indigo-900 border-indigo-500'
                    }`}
                >
                  WavLM Base Plus
                </button>
                <button
                  onClick={() => { setMode('wavlm_large'); setShowModelSelection(false); setAllSegmentResults({}); }}
                  className={`mode-btn ${mode === 'wavlm_large'
                    ? 'bg-rose-600 text-white shadow-sm border-rose-500'
                    : isDark ? 'bg-slate-700 text-rose-400 hover:text-rose-300 border-rose-500' : 'bg-white text-rose-600 hover:text-rose-700 border-rose-400'
                    }`}
                >
                  WavLM Large
                </button>
                {/* ── Ortak Test ── */}
                <button
                  onClick={() => { setMode('joint_test'); setShowModelSelection(false); setAllSegmentResults({}); setJointResults([]); setJointDone(false); }}
                  className={`mode-btn ${mode === 'joint_test'
                    ? 'bg-gradient-to-r from-amber-500 to-orange-500 text-white shadow-sm border-amber-400'
                    : isDark ? 'bg-slate-700 text-amber-400 hover:text-amber-300 border-amber-500' : 'bg-white text-amber-600 hover:text-amber-700 border-amber-400'
                    }`}
                >
                  {t('joint_test_button')}
                </button>
              </div>
            </div>
          )}

          {!audioFile && (
            <div className="w-full flex justify-center">
              <AudioInput onAudioReady={handleAudioReady} compact />
            </div>
          )}

          {audioFile && recordedUrl && !analysisResult && !isAnalyzing && !isJointTesting && !jointDone && (
            <div className="w-full max-w-2xl flex flex-col items-center animate-fadeIn">

              {mode === 'sentence_whole' && (
                <div className="w-full mb-6 flex justify-center">
                  <div className="px-6 py-2.5 rounded-2xl bg-gradient-to-r from-amber-500/20 to-orange-500/20 border border-amber-500/30 backdrop-blur-md flex items-center gap-3 shadow-sm">
                    <div className="w-2.5 h-2.5 rounded-full bg-amber-500 animate-pulse" />
                    <span className="text-sm font-black uppercase tracking-widest text-amber-600 dark:text-amber-400">
                      {t('majority_voting')} ({t('sentence_whole_badge')})
                    </span>
                  </div>
                </div>
              )}

              {mode === 'advanced_sentence' && (
                <div className="w-full mb-6 flex justify-center">
                  <div className="px-6 py-2.5 rounded-2xl bg-gradient-to-r from-red-500/20 to-rose-500/20 border border-red-500/30 backdrop-blur-md flex items-center gap-3 shadow-sm">
                    <div className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse" />
                    <span className="text-sm font-black uppercase tracking-widest text-red-600 dark:text-red-400">
                      {isTr ? "Gelişmiş Cümle Analizi" : "Advanced Sentence Analysis"}
                    </span>
                  </div>
                </div>
              )}

              {mode === 'hubert' && (
                <div className="w-full mb-6 flex justify-center">
                  <div className="px-6 py-2.5 rounded-2xl bg-gradient-to-r from-emerald-500/20 to-teal-500/20 border border-emerald-500/30 backdrop-blur-md flex items-center gap-3 shadow-sm">
                    <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
                    <span className="text-sm font-black uppercase tracking-widest text-emerald-600 dark:text-emerald-400">
                      SeaBenSea / HuBERT — HuggingFace Transformer
                    </span>
                  </div>
                </div>
              )}

              {mode === 'wav2vec2_turkish' && (
                <div className="w-full mb-6 flex justify-center">
                  <div className="px-6 py-2.5 rounded-2xl bg-gradient-to-r from-sky-500/20 to-blue-500/20 border border-sky-500/30 backdrop-blur-md flex items-center gap-3 shadow-sm">
                    <div className="w-2.5 h-2.5 rounded-full bg-sky-500 animate-pulse" />
                    <span className="text-sm font-black uppercase tracking-widest text-sky-600 dark:text-sky-400">
                      Sefa-Alper / Wav2Vec2 — Turkish Speech Emotion
                    </span>
                  </div>
                </div>
              )}

              {mode === 'superb_er' && (
                <div className="w-full mb-6 flex justify-center">
                  <div className="px-6 py-2.5 rounded-2xl bg-gradient-to-r from-teal-500/20 to-cyan-500/20 border border-teal-500/30 backdrop-blur-md flex items-center gap-3 shadow-sm">
                    <div className="w-2.5 h-2.5 rounded-full bg-teal-500 animate-pulse" />
                    <span className="text-sm font-black uppercase tracking-widest text-teal-600 dark:text-teal-400">
                      SUPERB / Wav2Vec2-Base-ER — 4-Class Emotion
                    </span>
                  </div>
                </div>
              )}

              {mode === 'sensevoice' && (
                <div className="w-full mb-6 flex justify-center">
                  <div className="px-6 py-2.5 rounded-2xl bg-gradient-to-r from-orange-500/20 to-amber-500/20 border border-orange-500/30 backdrop-blur-md flex items-center gap-3 shadow-sm">
                    <div className="w-2.5 h-2.5 rounded-full bg-orange-500 animate-pulse" />
                    <span className="text-sm font-black uppercase tracking-widest text-orange-600 dark:text-orange-400">
                      FunAudioLLM / SenseVoiceSmall — Alibaba
                    </span>
                  </div>
                </div>
              )}

              {mode === 'exhubert' && (
                <div className="w-full mb-6 flex justify-center">
                  <div className="px-6 py-2.5 rounded-2xl bg-gradient-to-r from-violet-500/20 to-purple-500/20 border border-violet-500/30 backdrop-blur-md flex items-center gap-3 shadow-sm">
                    <div className="w-2.5 h-2.5 rounded-full bg-violet-500 animate-pulse" />
                    <span className="text-sm font-black uppercase tracking-widest text-violet-600 dark:text-violet-400">
                      amiriparian / ExHuBERT — 6-Class Arousal-Valence
                    </span>
                  </div>
                </div>
              )}

              {mode === 'wavlm_large' && (
                <div className="w-full mb-6 flex justify-center">
                  <div className="px-6 py-2.5 rounded-2xl bg-gradient-to-r from-rose-500/20 to-pink-500/20 border border-rose-500/30 backdrop-blur-md flex items-center gap-3 shadow-sm">
                    <div className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-pulse" />
                    <span className="text-sm font-black uppercase tracking-widest text-rose-600 dark:text-rose-400">
                      3loi / WavLM Large — Odyssey 2024 · 8-Class
                    </span>
                  </div>
                </div>
              )}

              {/* Model Selection UI - Matrix Format */}
              {(mode === 'word' || (showModelSelection && mode !== 'sentence_whole' && mode !== 'advanced_sentence')) && (
                <div className={`w-full mt-6 mb-8 rounded-[2rem] border shadow-2xl animate-fadeIn overflow-hidden relative ${isDark ? 'border-slate-700/50 bg-slate-800/40 backdrop-blur-xl' : 'border-slate-200 bg-white'}`}>

                  {/* Glassmorphic overlay instead of solid gradient */}
                  <div className={`absolute inset-0 bg-gradient-to-b pointer-events-none ${isDark ? 'from-slate-700/20 to-transparent' : 'from-white/20 to-transparent'}`} />

                  {/* Quality Section */}
                  <div className="relative p-5 pb-4">
                    <p className={`text-xs font-bold uppercase tracking-widest mb-3 text-center ${isDark ? 'text-slate-500' : 'text-slate-500'}`}>
                      {t('training_quality')}
                    </p>
                    <div className="flex justify-center" style={{ gap: '16px' }}>
                      <button
                        onClick={() => setQualityMode('studio')}
                        style={{ padding: '10px 32px' }}
                        className={`text-sm font-bold transition-all duration-300 rounded-xl ${qualityMode === 'studio'
                          ? 'bg-gradient-to-br from-indigo-500 to-purple-500 text-white shadow-[0_4px_12px_rgba(99,102,241,0.3)] border-transparent'
                          : isDark
                            ? 'bg-transparent text-slate-300 border border-slate-700 hover:bg-slate-800'
                            : 'bg-transparent text-slate-700 border border-slate-300 hover:bg-slate-50'
                        }`}
                      >
                        {t('studio')}
                      </button>
                      <button
                        onClick={() => setQualityMode('robust')}
                        style={{ padding: '10px 32px' }}
                        className={`text-sm font-bold transition-all duration-300 rounded-xl ${qualityMode === 'robust'
                          ? 'bg-gradient-to-br from-purple-500 to-indigo-400 text-white shadow-[0_4px_12px_rgba(139,92,246,0.3)] border-transparent'
                          : isDark
                            ? 'bg-transparent text-slate-300 border border-slate-700 hover:bg-slate-800'
                            : 'bg-transparent text-slate-700 border border-slate-300 hover:bg-slate-50'
                        }`}
                      >
                        {t('noisy')}
                      </button>
                    </div>
                  </div>

                  {/* Divider */}
                  <div className={`border-t ${isDark ? 'border-slate-700/60' : 'border-slate-200'}`} />

                  {/* Model Grid */}
                  <div className="relative p-5 pt-4">
                    <p className={`text-xs font-bold uppercase tracking-widest mb-3 text-center ${isDark ? 'text-slate-500' : 'text-slate-500'}`}>
                      {t('analysis_engine')}
                    </p>
                    <div className="grid grid-cols-3 gap-2">
                      {BASE_MODELS.map((model) => (
                        <button
                          key={model.id}
                          onClick={() => setSelectedModel(model.id)}
                          className={`transition-all duration-200 rounded-lg text-xs sm:text-sm font-semibold px-2 py-2.5 ${selectedModel === model.id
                            ? 'bg-gradient-to-br from-indigo-500 to-purple-500 text-white shadow-[0_4px_15px_rgba(99,102,241,0.3)] border-transparent'
                            : isDark
                              ? 'bg-transparent text-slate-300 border border-slate-700 hover:bg-slate-800'
                              : 'bg-transparent text-slate-700 border border-slate-300 hover:bg-slate-50'
                          }`}
                        >
                          {model.name}
                        </button>
                      ))}
                    </div>

                    {/* Majority Voting Button */}
                    <button
                      onClick={() => setSelectedModel('majority_voting')}
                      className={`w-full transition-all duration-200 rounded-lg text-sm font-bold mt-2.5 px-2 py-3 ${selectedModel === 'majority_voting'
                        ? 'bg-gradient-to-br from-amber-500 to-red-500 text-white shadow-[0_4px_15px_rgba(245,158,11,0.35)] border-transparent'
                        : isDark
                          ? 'bg-transparent text-amber-400 border-2 border-dashed border-amber-500 hover:bg-amber-500/10'
                          : 'bg-transparent text-amber-600 border-2 border-dashed border-amber-500 hover:bg-amber-50'
                      }`}
                    >
                      {t('majority_voting')}
                    </button>
                  </div>
                </div>
              )}

              {/* Segmentation Preview UI */}
              {mode === 'sentence_segmented' && !showModelSelection && (
                <div className={`w-full mb-8 p-6 rounded-2xl border shadow-sm animate-fadeIn ${isDark ? 'bg-slate-800/50 border-slate-700' : 'bg-white border-slate-200'}`}>
                  {isSegmenting ? (
                    <div className="flex flex-col items-center justify-center p-4">
                      <div className="w-8 h-8 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mb-2" />
                      <p className="text-sm text-slate-600 dark:text-slate-300 font-medium">{t('segmenting')}</p>
                    </div>
                  ) : hasAnyResults ? (
                    <div className="flex flex-col items-center">
                      {/* Engine Tabs */}
                      <div className="flex justify-center mb-5" style={{ gap: '8px' }}>
                        {[
                          { id: 'vad' as const, label: 'VAD', gradient: 'linear-gradient(135deg, #6366f1, #818cf8)' },
                          { id: 'vosk' as const, label: 'VOSK', gradient: 'linear-gradient(135deg, #10b981, #34d399)' },
                          { id: 'whisperx' as const, label: 'WHISPERX', gradient: 'linear-gradient(135deg, #f59e0b, #fbbf24)' },
                        ].map((tab) => {
                          const tabData = allSegmentResults[tab.id];
                          const count = tabData?.segments?.length ?? 0;
                          const hasError = !!tabData?.error;
                          return (
                            <button
                              key={tab.id}
                              onClick={() => setActiveSegTab(tab.id)}
                              className={`text-sm font-bold transition-all duration-300 rounded-xl relative flex flex-col items-center justify-center leading-tight ${
                                activeSegTab === tab.id
                                  ? 'text-white shadow-[0_4px_12px_rgba(0,0,0,0.15)] border-transparent'
                                  : hasError
                                    ? `bg-transparent border ${isDark ? 'text-red-400 border-red-500/50 hover:bg-red-500/10' : 'text-red-500 border-red-300 hover:bg-red-50'}`
                                    : `bg-transparent border ${isDark ? 'text-slate-300 border-slate-700 hover:bg-slate-800' : 'text-slate-600 border-slate-200 hover:bg-slate-50'}`
                              }`}
                              style={{ padding: '8px 24px', ...(activeSegTab === tab.id ? { background: tab.gradient } : {}) }}
                            >
                              <span>
                                {tab.label}
                                <span className="ml-1.5 text-[10px] opacity-75 inline-block">
                                  ({hasError ? '✗' : count})
                                </span>
                              </span>
                              {!hasError && tabData?.elapsed !== undefined && (
                                <span className="text-[10px] opacity-70 font-mono font-medium -mt-0.5">
                                  {tabData.elapsed}s
                                </span>
                              )}
                            </button>
                          );
                        })}
                      </div>

                      {/* Active Tab Content */}
                      {segError ? (
                        <div className="text-center p-4">
                          <p className="text-red-500 dark:text-red-400 text-sm font-medium">⚠️ {segError}</p>
                        </div>
                      ) : sentenceSegments && sentenceSegments.length > 0 ? (
                        <>
                          <label className="block text-sm font-semibold text-slate-600 dark:text-slate-300 mb-4 text-center">
                            {t('segmented_words')} ({sentenceSegments.length} {t('pieces')})
                          </label>
                          <div className="flex flex-wrap gap-2 justify-center px-4">
                            {sentenceSegments.map((seg, i) => {
                              const EMOTION_COLORS: Record<string, string> = {
                                happy: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-300',
                                sad: 'bg-blue-100 text-blue-700 dark:bg-blue-500/20 dark:text-blue-300',
                                angry: 'bg-red-100 text-red-700 dark:bg-red-500/20 dark:text-red-300',
                                calm: 'bg-purple-100 text-purple-700 dark:bg-purple-500/20 dark:text-purple-300',
                                neutral: 'bg-white border border-slate-200 text-slate-700 shadow-sm dark:bg-slate-800 dark:border-transparent dark:text-slate-300'
                              };
                              const emotionKey = seg.emotion || 'neutral';
                              const colorClass = EMOTION_COLORS[emotionKey] || 'bg-white border border-slate-200 text-slate-700 shadow-sm dark:bg-slate-800 dark:border-transparent dark:text-slate-300';

                              return (
                                <button
                                  key={i}
                                  className={`rounded-lg text-sm font-medium hover:opacity-80 transition-opacity shadow-sm flex items-center gap-1.5 ${colorClass}`}
                                  style={{ padding: '6px 14px' }}
                                  onClick={() => {
                                    if (audioElementRef.current) {
                                      audioElementRef.current.currentTime = seg.start;
                                      audioElementRef.current.play();
                                      setIsPlaying(true);
                                      setTimeout(() => {
                                        if (audioElementRef.current) {
                                          audioElementRef.current.pause();
                                        }
                                        setIsPlaying(false);
                                      }, (seg.end - seg.start) * 1000);
                                    }
                                  }}
                                >
                                  <span>{seg.word ? `"${seg.word}"` : `${t('word_label')} ${i + 1}`}</span>
                                  <span className="text-[10px] opacity-75">({seg.start.toFixed(1)}s - {seg.end.toFixed(1)}s)</span>
                                  {seg.emotion && seg.emotion !== '?' && seg.emotion !== 'error' && (
                                    <span className="ml-0.5 text-[10px] font-bold uppercase tracking-wider">{t(seg.emotion.toLowerCase())}</span>
                                  )}
                                </button>
                              );
                            })}
                          </div>
                        </>
                      ) : (
                        <p className="text-center text-slate-400 text-sm">{t('segmented_words')}: 0 {t('pieces')}</p>
                      )}
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center p-2">
                      <p className={`text-center mb-4 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>{t('segment_prompt')}</p>
                      <button
                        onClick={() => audioFile && handleSegmentation(audioFile)}
                        className="transition-all duration-300 rounded-xl text-sm font-bold"
                        style={{
                          padding: '10px 32px',
                          background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                          color: 'white',
                          boxShadow: '0 4px 15px rgba(99,102,241,0.3)',
                        }}
                      >
                        {t('segment_button')}
                      </button>
                    </div>
                  )}
                </div>
              )}

              <AudioPlayer
                mode="preview"
                analysisMode={mode === 'word' ? 'word' : 'sentence'}
                selectedModelName={
                  mode === 'joint_test' ? `⚡ ${t('joint_test_button')} — ${JOINT_MODELS.length} Model`
                    : mode === 'hubert' ? 'SeaBenSea/HuBERT (HuggingFace)'
                      : mode === 'exhubert' ? 'amiriparian/ExHuBERT'
                        : mode === 'wav2vec2_turkish' ? 'Sefa-Alper/Wav2Vec2 Turkish'
                          : mode === 'superb_er' ? 'SUPERB / Wav2Vec2-Base-ER'
                            : mode === 'sensevoice' ? 'SenseVoiceSmall (Alibaba)'
                              : mode === 'wavlm_base_plus' ? 'harritaylor / WavLM Base Plus'
                                : mode === 'wavlm_large' ? '3loi / WavLM Large (Odyssey 2024)'
                                  : mode === 'sentence_whole' ? t('sentence_whole_btn')
                                    : mode === 'advanced_sentence' ? (isTr ? 'Gelişmiş Cümle Analizi' : 'Advanced Sentence Analysis')
                                      : activeModelName
                }
                recordedUrl={recordedUrl}
                showAnalyzeButton={mode === 'word' || mode === 'sentence_whole' || mode === 'advanced_sentence' || mode === 'hubert' || mode === 'exhubert' || mode === 'wav2vec2_turkish' || mode === 'superb_er' || mode === 'sensevoice' || mode === 'wavlm_base_plus' || mode === 'wavlm_large' || mode === 'joint_test' || showModelSelection}
                levels={savedLevels}
                isPlaying={isPlaying}
                playProgress={playProgress}
                playbackRate={playbackRate}
                onTogglePlay={tooglePlayPause}
                onSpeedChange={handleSpeedChange}
                onAnalyze={mode === 'joint_test' ? handleJointTest : handleAnalyze}
                onBack={reset}
                isSpeedMenuOpen={isSpeedMenuOpen}
                setIsSpeedMenuOpen={setIsSpeedMenuOpen}
                // Props not needed for simple preview but required by component signature:
                isRecording={false}
                recordingTime={currentTime > 0 ? currentTime : duration}
                duration={duration}
                currentTime={currentTime}
                onStartRecording={() => { }}
                onStopRecording={() => { }}
              />

              {/* İleri button - below AudioPlayer */}
              {mode === 'sentence_segmented' && !showModelSelection && sentenceSegments && (
                <button
                  onClick={() => setShowModelSelection(true)}
                  className="transition-all duration-300 rounded-xl text-base font-bold hover:scale-[1.02]"
                  style={{
                    marginTop: '24px',
                    padding: '14px 48px',
                    background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                    color: 'white',
                    boxShadow: '0 4px 15px rgba(99,102,241,0.3)',
                  }}
                >
                  {t('next_select')}
                </button>
              )}
            </div>
          )}

          {isAnalyzing && (
            <div className="flex flex-col items-center justify-center w-full h-full animate-fadeIn py-20 z-10">
              <div className="relative w-24 h-24 mb-6">
                <div className="absolute inset-0 border-4 border-indigo-200/20 dark:border-indigo-500/20 rounded-full" />
                <div className="absolute inset-0 border-4 border-indigo-600 dark:border-indigo-400 border-t-transparent rounded-full animate-[spin_1s_linear_infinite]" />
                <div className="absolute inset-2 border-4 border-purple-600 dark:border-purple-400 border-b-transparent rounded-full animate-[spin_1.5s_linear_infinite_reverse]" />
              </div>
              <p className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-500 to-purple-500 animate-pulse">
                {t('analyzing')}
              </p>
              <p className="text-sm font-medium text-slate-500 dark:text-slate-400 mt-3 tracking-widest uppercase">
                {t('engine_used')}: {
                  mode === 'joint_test' ? `⚡ Joint Test — ${JOINT_MODELS.length} Model`
                    : mode === 'hubert' ? 'SeaBenSea/HuBERT (HuggingFace)'
                      : mode === 'exhubert' ? 'amiriparian/ExHuBERT'
                        : mode === 'wav2vec2_turkish' ? 'Sefa-Alper/Wav2Vec2 Turkish'
                          : mode === 'superb_er' ? 'SUPERB / Wav2Vec2-Base-ER'
                            : mode === 'sensevoice' ? 'FunAudioLLM/SenseVoiceSmall (Alibaba)'
                              : mode === 'wavlm_base_plus' ? 'harritaylor / WavLM Base Plus'
                                : mode === 'wavlm_large' ? '3loi / WavLM Large (Odyssey 2024)'
                                  : mode === 'advanced_sentence' ? (isTr ? 'Gelişmiş Cümle Analizi' : 'Advanced Sentence Analysis')
                                    : mode === 'sentence_whole' ? t('sentence_whole_engine')
                                      : activeModelName
                }
              </p>
            </div>
          )}

          {analysisResult && (
            <div className="w-full animate-fadeIn">
              <Result
                result={analysisResult}
                onBack={reset}
                audioUrl={recordedUrl || undefined}
              />
            </div>
          )}

          {/* ── Joint Test: Loading ── */}
          {isJointTesting && (
            <div className="flex flex-col items-center justify-center w-full py-16 animate-fadeIn gap-6">
              <div className="relative w-20 h-20">
                <div className="absolute inset-0 border-4 border-amber-200/20 rounded-full" />
                <div className="absolute inset-0 border-4 border-amber-500 border-t-transparent rounded-full animate-[spin_1s_linear_infinite]" />
                <div className="absolute inset-2 border-4 border-orange-400 border-b-transparent rounded-full animate-[spin_1.5s_linear_infinite_reverse]" />
              </div>
              <p className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-amber-400 to-orange-500 animate-pulse">
                ⚡ {JOINT_MODELS.length} {t('joint_test_analyzing')}
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3 w-full max-w-4xl px-4">
                {jointResults.map((r, i) => (
                  <div key={i} className={`flex items-center gap-3 px-4 py-3 rounded-2xl border transition-all duration-500 ${r.status === 'done' ? 'border-emerald-500/40 bg-emerald-500/10' :
                      r.status === 'error' ? 'border-red-500/40 bg-red-500/10' :
                        'border-white/10 bg-white/5'
                    }`}>
                    <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: r.status === 'done' ? '#10b981' : r.status === 'error' ? '#ef4444' : '#94a3b8' }} />
                    <div className="min-w-0">
                      <p className="text-xs font-bold truncate opacity-70">
                        {r.name === 'Gelişmiş Cümle Analizi'
                          ? (isTr ? 'Gelişmiş Cümle Analizi' : 'Advanced Sentence Analysis')
                          : (r.name || r.model)}
                      </p>
                      {r.status === 'done' && <p className="text-sm font-black capitalize" style={{ color: r.color }}>{r.emotion}</p>}
                      {r.status === 'error' && <p className="text-xs text-red-400">{t('joint_test_error_label')}</p>}
                      {r.status === 'loading' && <p className="text-xs opacity-40">{t('joint_test_waiting')}</p>}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── Joint Test: Results ── */}
          {jointDone && !isJointTesting && (
            <div className="w-full animate-fadeIn px-4 md:px-8 py-6">
              {/* Header */}
              <div className="flex items-center justify-between mb-8 max-w-5xl mx-auto">
                <button
                  onClick={reset}
                  className={`w-11 h-11 flex items-center justify-center rounded-full transition-all hover:scale-110 ${isDark ? 'text-slate-400 hover:text-white' : 'text-slate-500 hover:text-slate-900'}`}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" /></svg>
                </button>
                <div className="flex items-center gap-3">
                  <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
                  <span className={`text-sm font-black uppercase tracking-widest ${isDark ? 'text-amber-400' : 'text-amber-600'}`}>{t('joint_test_results_title')}</span>
                </div>
                <div className="w-11" />
              </div>

              {/* Results Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 max-w-5xl mx-auto">
                {jointResults.map((r, i) => (
                  <div
                    key={i}
                    className={`relative flex flex-col items-center text-center gap-3 p-6 rounded-2xl border transition-all duration-300 ${r.status === 'error'
                        ? isDark ? 'bg-red-500/5 border-red-500/20' : 'bg-red-50 border-red-200'
                        : isDark ? 'bg-slate-900/60 border-white/10' : 'bg-white/70 border-slate-200/60'
                      }`}
                    style={{ backdropFilter: 'blur(12px)' }}
                  >
                    {/* Color accent top bar */}
                    <div className="absolute top-0 left-6 right-6 h-0.5 rounded-full" style={{ background: r.color }} />

                    <p className={`text-xs font-black uppercase tracking-wider ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                      {r.name === 'Gelişmiş Cümle Analizi'
                        ? (isTr ? 'Gelişmiş Cümle Analizi' : 'Advanced Sentence Analysis')
                        : (r.name || r.model)}
                    </p>

                    {r.status === 'done' ? (
                      <>
                        <p className="text-2xl font-black capitalize" style={{ color: r.color }}>
                          {r.emotion}
                        </p>
                        <div className="w-full">
                          <div className="flex justify-between text-xs font-bold mb-1 opacity-60">
                            <span>{t('joint_test_confidence')}</span>
                            <span>{r.confidence.toFixed(1)}%</span>
                          </div>
                          <div className={`h-1.5 w-full rounded-full ${isDark ? 'bg-slate-700' : 'bg-slate-200'}`}>
                            <div className="h-full rounded-full transition-all duration-700" style={{ width: `${Math.min(r.confidence, 100)}%`, background: r.color }} />
                          </div>
                        </div>
                      </>
                    ) : (
                      <p className="text-sm font-bold text-red-400">{r.error || t('joint_test_error_occurred')}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {recordedUrl && (
            <audio
              ref={audioElementRef}
              src={recordedUrl}
              onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
              onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
              onEnded={() => { setIsPlaying(false); setPlayProgress(0); setCurrentTime(0); }}
              className="hidden"
            />
          )}

        </div>
      </div>
    </div>
  );
};

export default Hero;
