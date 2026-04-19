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
import InteractiveBackground from "./InteractiveBackground";

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
  const { t } = useTranslation();
  const { isDark } = useTheme();

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
  const [mode, setMode] = useState<'word' | 'sentence_segmented' | 'sentence_whole'>('word');
  const [sttEngine, setSttEngine] = useState<'vad' | 'vosk' | 'whisperx'>('vad');

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

    if (mode === 'sentence_segmented') {
      // Ayrıştırma için butona basılmasını bekle
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
          const resp = await axios.post(`http://localhost:5000/segment-sentence`, fd, {
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
          const resp = await axios.post(`http://localhost:5000/transcribe`, fd, {
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
          const resp = await axios.post(`http://localhost:5000/transcribe`, fd, {
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
      if (selectedModel === 'majority_voting') {
        endpoint = '/analyze_voting';
      } else if (mode === 'word') {
        endpoint = '/predict';
      } else if (mode === 'sentence_segmented') {
        endpoint = '/predict-sentence';
      } else {
        endpoint = '/api/predict_sentence_whole';
      }
      const response = await axios.post(`http://localhost:5000${endpoint}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      // Parse response based on endpoint type
      let emotion, confidence, all_scores;

      if (selectedModel === 'majority_voting') {
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
  };

  return (
    <div className={clsx(
        "relative w-full flex-grow flex flex-col items-center font-sans transition-colors duration-500",
        isDark ? "bg-[#0b0f19] text-white" : "bg-gray-50 text-slate-900",
        analysisResult
            ? "justify-start pt-24 pb-12 overflow-x-hidden"
            : "justify-center overflow-hidden"
    )}>
      
      <InteractiveBackground />

      <div className="relative z-10 w-full max-w-6xl px-6 flex flex-col items-center py-20 mb-10">

        <h1 className={`font-outfit text-5xl md:text-7xl font-black mb-4 py-2 leading-tight text-center tracking-tighter transition-all duration-700 ease-[cubic-bezier(0.16,1,0.3,1)] ${analysisResult ? "scale-75 opacity-0 h-0" : "opacity-100"}`}>
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-fuchsia-500 to-rose-500 drop-shadow-sm">
            {t('discover_your_voice')}
          </span>
        </h1>

        {!analysisResult && (
          <p className={`text-base md:text-lg font-medium mb-8 text-center max-w-2xl leading-relaxed tracking-wide ${isDark ? "text-indigo-100/60" : "text-slate-500/90"} animate-slideUpFade`} style={{ animationDelay: '0.05s' }}>
            <Trans i18nKey="experimental_subtitle">
              Burada <strong className="font-semibold text-indigo-400 dark:text-indigo-300">deneysel modeller</strong> mevcuttur — farklı yapay zeka motorlarını test edebilir ve sonuçları karşılaştırabilirsiniz.
            </Trans>
          </p>
        )}

        <div className={`
          relative w-full backdrop-blur-[40px] shadow-2xl transition-all duration-1000 ease-[cubic-bezier(0.16,1,0.3,1)]
          flex flex-col items-center justify-center border
          ${isDark ? "bg-[#0f172a]/70 border-white/10 shadow-[0_0_100px_rgba(99,102,241,0.15)]" : "bg-white/70 border-indigo-100/80 shadow-[0_0_100px_rgba(99,102,241,0.1)]"}
          ${analysisResult ? "max-w-[100vw] sm:max-w-[98vw] lg:max-w-[1600px] min-h-[85vh] p-3 md:p-8 lg:p-10 overflow-visible rounded-3xl md:rounded-[2.5rem] mx-auto border-indigo-500/20" : "max-w-5xl min-h-[320px] p-6 md:p-10 rounded-[3rem]"}
        `}
          style={!analysisResult ? { padding: '32px 40px 48px 40px' } : { marginTop: '80px' }}
        >

          {/* Subtle Inner Glow */}
          {!analysisResult && (
             <div className="absolute inset-0 rounded-[3rem] pointer-events-none shadow-[inset_0_0_60px_rgba(255,255,255,0.05)]" />
          )}
          {/* Mode Switcher */}
          {!analysisResult && (
            <div className="flex flex-wrap justify-center relative z-20" style={{ gap: '12px', marginBottom: '24px' }}>
              <button
                onClick={() => { setMode('word'); setShowModelSelection(true); }}
                className={`rounded-xl text-sm font-bold transition-all duration-300 border-2 ${mode === 'word'
                  ? 'bg-white dark:bg-slate-600 text-indigo-600 dark:text-white shadow-lg border-indigo-400 dark:border-indigo-500'
                  : 'bg-slate-200 dark:bg-slate-700 text-slate-500 dark:text-slate-400 hover:text-slate-700 border-slate-300 dark:border-slate-600'
                  }`}
                style={{ padding: '10px 24px' }}
              >
                {t('mode_word')}
              </button>
              <button
                onClick={() => { setMode('sentence_segmented'); setShowModelSelection(false); setAllSegmentResults({}); }}
                className={`rounded-xl text-sm font-bold transition-all duration-300 border-2 ${mode === 'sentence_segmented'
                  ? 'bg-white dark:bg-slate-600 text-indigo-600 dark:text-white shadow-lg border-indigo-400 dark:border-indigo-500'
                  : 'bg-slate-200 dark:bg-slate-700 text-slate-500 dark:text-slate-400 hover:text-slate-700 border-slate-300 dark:border-slate-600'
                  }`}
                style={{ padding: '10px 24px' }}
              >
                {t('mode_sentence_segmented')}
              </button>
              <button
                onClick={() => { setMode('sentence_whole'); setShowModelSelection(true); setAllSegmentResults({}); }}
                className={`rounded-xl text-sm font-bold transition-all duration-300 border-2 ${mode === 'sentence_whole'
                  ? 'bg-white dark:bg-slate-600 text-indigo-600 dark:text-white shadow-lg border-indigo-400 dark:border-indigo-500'
                  : 'bg-slate-200 dark:bg-slate-700 text-slate-500 dark:text-slate-400 hover:text-slate-700 border-slate-300 dark:border-slate-600'
                  }`}
                style={{ padding: '10px 24px' }}
              >
                {t('mode_sentence_whole')}
              </button>
            </div>
          )}

          {!audioFile && (
            <AudioInput onAudioReady={handleAudioReady} />
          )}

          {audioFile && recordedUrl && !analysisResult && !isAnalyzing && (
            <div className="w-full max-w-2xl flex flex-col items-center animate-fadeIn">
              
              {mode === 'sentence_whole' && (
                <div className="w-full mb-6 flex justify-center">
                  <div className="px-6 py-2.5 rounded-2xl bg-gradient-to-r from-amber-500/20 to-orange-500/20 border border-amber-500/30 backdrop-blur-md flex items-center gap-3 shadow-sm">
                    <div className="w-2.5 h-2.5 rounded-full bg-amber-500 animate-pulse" />
                    <span className="text-sm font-black uppercase tracking-widest text-amber-600 dark:text-amber-400">
                      {t('majority_voting')} (10-Model Optimized Ensemble)
                    </span>
                  </div>
                </div>
              )}

              {/* Model Selection UI - Matrix Format */}
              {(mode === 'word' || (showModelSelection && mode !== 'sentence_whole')) && (
                <div className="w-full mb-8 rounded-[2rem] border border-white/40 dark:border-slate-700/50 shadow-2xl animate-fadeIn overflow-hidden relative bg-white/40 dark:bg-slate-800/40 backdrop-blur-xl">

                  {/* Glassmorphic overlay instead of solid gradient */}
                  <div className="absolute inset-0 bg-gradient-to-b from-white/20 to-transparent dark:from-slate-700/20 pointer-events-none" />

                  {/* Quality Section */}
                  <div className="relative p-5 pb-4">
                    <p className="text-xs font-bold uppercase tracking-widest text-slate-400 dark:text-slate-500 mb-3 text-center">
                      {t('training_quality')}
                    </p>
                    <div className="flex justify-center" style={{ gap: '16px' }}>
                      <button
                        onClick={() => setQualityMode('studio')}
                        className="text-sm font-bold transition-all duration-300 rounded-xl"
                        style={qualityMode === 'studio'
                          ? { padding: '10px 32px', background: 'linear-gradient(135deg, #6366f1, #818cf8)', color: 'white', boxShadow: '0 4px 12px rgba(99,102,241,0.3)' }
                          : { padding: '10px 32px', color: '#94a3b8', border: '1px solid #e2e8f0' }
                        }
                      >
                        {t('studio')}
                      </button>
                      <button
                        onClick={() => setQualityMode('robust')}
                        className="text-sm font-bold transition-all duration-300 rounded-xl"
                        style={qualityMode === 'robust'
                          ? { padding: '10px 32px', background: 'linear-gradient(135deg, #8b5cf6, #a78bfa)', color: 'white', boxShadow: '0 4px 12px rgba(139,92,246,0.3)' }
                          : { padding: '10px 32px', color: '#94a3b8', border: '1px solid #e2e8f0' }
                        }
                      >
                        {t('noisy')}
                      </button>
                    </div>
                  </div>

                  {/* Divider */}
                  <div className="border-t border-slate-200/60 dark:border-slate-700/60" />

                  {/* Model Grid */}
                  <div className="relative p-5 pt-4">
                    <p className="text-xs font-bold uppercase tracking-widest text-slate-400 dark:text-slate-500 mb-3 text-center">
                      {t('analysis_engine')}
                    </p>
                    <div className="grid grid-cols-3 gap-2">
                      {BASE_MODELS.map((model) => (
                        <button
                          key={model.id}
                          onClick={() => setSelectedModel(model.id)}
                          className="transition-all duration-200 rounded-lg text-xs sm:text-sm font-semibold"
                          style={selectedModel === model.id
                            ? { padding: '10px 8px', background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', color: 'white', boxShadow: '0 4px 15px rgba(99,102,241,0.3)' }
                            : { padding: '10px 8px', background: 'transparent', color: '#64748b', border: '1px solid #e2e8f0' }
                          }
                        >
                          {model.name}
                        </button>
                      ))}
                    </div>

                    {/* Majority Voting Button */}
                    <button
                      onClick={() => setSelectedModel('majority_voting')}
                      className="w-full transition-all duration-200 rounded-lg text-sm font-bold"
                      style={selectedModel === 'majority_voting'
                        ? { marginTop: '10px', padding: '12px 8px', background: 'linear-gradient(135deg, #f59e0b, #ef4444)', color: 'white', boxShadow: '0 4px 15px rgba(245,158,11,0.35)' }
                        : { marginTop: '10px', padding: '12px 8px', background: 'transparent', color: '#f59e0b', border: '2px dashed #f59e0b' }
                      }
                    >
                      {t('majority_voting')}
                    </button>
                  </div>
                </div>
              )}

              {/* Segmentation Preview UI */}
              {mode === 'sentence_segmented' && !showModelSelection && (
                <div className="w-full mb-8 p-6 bg-white dark:bg-slate-800/50 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm animate-fadeIn">
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
                          { id: 'vad' as const, label: t('stt_vad'), gradient: 'linear-gradient(135deg, #6366f1, #818cf8)' },
                          { id: 'vosk' as const, label: t('stt_vosk'), gradient: 'linear-gradient(135deg, #10b981, #34d399)' },
                          { id: 'whisperx' as const, label: t('stt_whisperx'), gradient: 'linear-gradient(135deg, #f59e0b, #fbbf24)' },
                        ].map((tab) => {
                          const tabData = allSegmentResults[tab.id];
                          const count = tabData?.segments?.length ?? 0;
                          const hasError = !!tabData?.error;
                          return (
                            <button
                              key={tab.id}
                              onClick={() => setActiveSegTab(tab.id)}
                              className="text-sm font-bold transition-all duration-300 rounded-xl relative flex flex-col items-center justify-center leading-tight"
                              style={activeSegTab === tab.id
                                ? {
                                  padding: '6px 20px',
                                  background: tab.gradient,
                                  color: 'white',
                                  boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
                                }
                                : {
                                  padding: '6px 20px',
                                  color: hasError ? '#ef4444' : '#94a3b8',
                                  border: `1px solid ${hasError ? '#fca5a5' : '#e2e8f0'}`
                                }
                              }
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
                          <div className="flex flex-wrap gap-2 justify-center">
                            {sentenceSegments.map((seg, i) => {
                              const EMOTION_COLORS: Record<string, string> = {
                                happy: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-300',
                                sad: 'bg-blue-100 text-blue-700 dark:bg-blue-500/20 dark:text-blue-300',
                                angry: 'bg-red-100 text-red-700 dark:bg-red-500/20 dark:text-red-300',
                                calm: 'bg-purple-100 text-purple-700 dark:bg-purple-500/20 dark:text-purple-300',
                                neutral: 'bg-slate-100 text-slate-700 dark:bg-slate-500/20 dark:text-slate-300'
                              };
                              const emotionKey = seg.emotion || 'neutral';
                              const colorClass = EMOTION_COLORS[emotionKey] || 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300';

                              return (
                                <button
                                  key={i}
                                  className={`px-3 py-1.5 rounded-lg text-sm font-medium hover:opacity-80 transition-opacity shadow-sm flex items-center gap-1.5 ${colorClass}`}
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
                      <p className="text-center text-slate-500 dark:text-slate-400 mb-4">{t('segment_prompt')}</p>
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
                selectedModelName={mode === 'sentence_whole' ? 'Experimental Ensemble' : activeModelName}
                recordedUrl={recordedUrl}
                showAnalyzeButton={mode === 'word' || mode === 'sentence_whole' || showModelSelection}
                levels={savedLevels}
                isPlaying={isPlaying}
                playProgress={playProgress}
                playbackRate={playbackRate}
                onTogglePlay={tooglePlayPause}
                onSpeedChange={handleSpeedChange}
                onAnalyze={handleAnalyze}
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
                {t('engine_used')}: {activeModelName}
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