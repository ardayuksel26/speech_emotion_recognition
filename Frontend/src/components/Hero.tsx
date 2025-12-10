import { useState, useRef, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { useTheme } from "../context/ThemeContext";
import AudioInput from "./AudioInput/AudioInput"; // Using the new component
import AudioPlayer from "./AudioPlayer";
import Result from "./Result";
import axios from "axios";
import { convertFileToWav } from "../utils/audioUtils";
import { AnalysisResult } from "../types";

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

  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [savedLevels, setSavedLevels] = useState<number[]>([]);

  // Audio Player State
  const [isPlaying, setIsPlaying] = useState(false);
  const [playProgress, setPlayProgress] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [isSpeedMenuOpen, setIsSpeedMenuOpen] = useState(false);

  const audioElementRef = useRef<HTMLAudioElement>(null);
  const playbackAnimationRef = useRef<number>();

  const handleAudioReady = async (file: File) => {
    setAudioFile(file);
    const url = URL.createObjectURL(file);
    setRecordedUrl(url);

    // Generate levels for preview
    const levels = await generateWaveLevels(file, 60);
    setSavedLevels(levels);
  };

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
      // Default strategy
      formData.append("aggregation_strategy", "weighted_average");

      // 1. Start Analysis Job
      const response = await axios.post("http://localhost:8000/api/analyze-sentence", formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const { job_id } = response.data;

      // 2. Poll for Results
      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await axios.get(`http://localhost:8000/api/status/${job_id}`);
          const { status, result, error } = statusRes.data;

          if (status === 'completed' && result) {
            clearInterval(pollInterval);
            // Map Backend Result to Frontend AnalysisResult
            setAnalysisResult({
              dominant_emotion: result.primary_emotion,
              emotions: result.probabilities,
              confidence: result.confidence,
              word_timestamps: result.word_predictions?.map((w: any) => ({
                word: `Word ${w.word_index}`, // Backend doesn't give text yet, just index/time
                start: w.start_time,
                end: w.end_time,
                emotion: w.emotion,
                confidence: w.confidence
              }))
            });
            setIsAnalyzing(false);
          } else if (status === 'failed' || status === 'timeout') {
            clearInterval(pollInterval);
            setIsAnalyzing(false);
            alert(`Analysis failed: ${error?.message || 'Unknown error'}`);
          }
          // If 'queued' or 'processing', continue polling
        } catch (err) {
          console.error("Polling error", err);
          clearInterval(pollInterval);
          setIsAnalyzing(false);
          alert("Error checking analysis status");
        }
      }, 1000);

    } catch (error: any) {
      console.error("Analysis Request Failed:", error);
      setIsAnalyzing(false);
      let msg = t('analysis_failed') || "Analiz hatası.";
      if (error.response?.data?.detail) {
        const detail = error.response.data.detail;
        msg += typeof detail === 'object' ? ` (${detail.message || JSON.stringify(detail)})` : ` (${detail})`;
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
  };

  return (
    <div className={`relative w-full flex flex-col items-center justify-center overflow-hidden font-sans transition-colors duration-300 ${isDark ? "bg-slate-900" : "bg-gray-100"}`}>
      <div className="relative z-10 w-full max-w-5xl px-6 flex flex-col items-center py-20">



        <h1 className={`text-4xl md:text-6xl font-extrabold mb-10 ${isDark ? "text-white" : "text-gray-800"}`}>
          {t('discover_your_voice')}
        </h1>

        <div className={`w-full backdrop-blur-xl rounded-2xl shadow-2xl p-8 transition-all min-h-[400px] flex flex-col items-center justify-center ${isDark ? "bg-slate-800/40 border border-white/10" : "bg-white/80 border border-gray-200"}`}>

          {!audioFile && (
            <AudioInput onAudioReady={handleAudioReady} />
          )}

          {audioFile && recordedUrl && !analysisResult && !isAnalyzing && (
            <AudioPlayer
              mode="preview"
              recordedUrl={recordedUrl}
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
              recordingTime={0}
              onStartRecording={() => { }}
              onStopRecording={() => { }}
            />
          )}

          {isAnalyzing && (
            <div className="flex flex-col items-center animate-pulse">
              <div className="w-16 h-16 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mb-4" />
              <p className="text-xl font-medium text-indigo-500">{t('analyzing')}</p>
            </div>
          )}

          {analysisResult && (
            <Result
              result={analysisResult}
              onBack={reset}
              audioUrl={recordedUrl || undefined}
            />
          )}

          {recordedUrl && (
            <audio
              ref={audioElementRef}
              src={recordedUrl}
              onEnded={() => { setIsPlaying(false); setPlayProgress(0); }}
              className="hidden"
            />
          )}

        </div>
      </div>
    </div>
  );
};

export default Hero;