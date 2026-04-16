import { useState } from "react";
import { useTranslation, Trans } from "react-i18next";
import { useTheme } from "../context/ThemeContext";
import { clsx } from "clsx";
import AudioInput from "./AudioInput/AudioInput";
import Result from "./Result";
import axios from "axios";
import { convertFileToWav } from "../utils/audioUtils";
import { AnalysisResult } from "../types";
import InteractiveBackground from "./InteractiveBackground";

type SegmentItem = { start: number; end: number; word?: string; emotion?: string; confidence?: number };

const MainHero = () => {
  const { t } = useTranslation();
  const { isDark } = useTheme();

  const [recordedUrl, setRecordedUrl] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isSegmenting, setIsSegmenting] = useState(false);

  /* ── Ses hazır → Vosk segmentasyon + otomatik analiz ── */
  const handleAudioReady = async (file: File) => {
    const url = URL.createObjectURL(file);
    setRecordedUrl(url);
    await runSegmentAndAnalyze(file);
  };

  const runSegmentAndAnalyze = async (file: File) => {
    // Adım 1: Vosk segmentasyonu
    setIsSegmenting(true);
    setAnalysisResult(null);

    let segs: SegmentItem[] = [];
    try {
      const wavBlob = await convertFileToWav(file);
      const fd = new FormData();
      fd.append("file", wavBlob, "converted_audio.wav");
      fd.append("stt_engine", "vosk");
      fd.append("model_type", "catboost");
      const resp = await axios.post("http://localhost:5000/transcribe", fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      segs = (resp.data.words || []).map((w: any) => ({
        start: w.start,
        end: w.end,
        word: w.word,
        emotion: w.emotion,
        confidence: typeof w.confidence === 'number' ? w.confidence : undefined,
      }));
    } catch (err: any) {
      console.error("Segmentation failed:", err);
      // Segmentasyon başarısız olsa da analiz devam eder
    } finally {
      setIsSegmenting(false);
    }

    // Adım 2: Majority Voting analiz (studio)
    setIsAnalyzing(true);
    try {
      const wavBlob = await convertFileToWav(file);
      const formData = new FormData();
      formData.append("file", wavBlob, "converted_audio.wav");
      formData.append("quality", "studio");

      const response = await axios.post("http://localhost:5000/analyze_voting", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const { emotion, confidence, all_scores } = response.data;

      let confidenceValue = 0;
      if (typeof confidence === "string") confidenceValue = parseFloat(confidence.replace("%", ""));
      else if (typeof confidence === "number") confidenceValue = confidence;
      const normalizedConfidence = confidenceValue / 100;

      const emotionsMap: { [key: string]: number; happy: number; sad: number; angry: number; calm: number } = {
        happy: 0, sad: 0, angry: 0, calm: 0,
      };
      if (all_scores) {
        Object.entries(all_scores).forEach(([key, val]) => {
          emotionsMap[key.toLowerCase()] = (val as number) / 100;
        });
      } else {
        emotionsMap[emotion.toLowerCase()] = normalizedConfidence;
      }
      const totalSum = Object.values(emotionsMap).reduce((a, b) => a + b, 0);
      if (totalSum > 0) Object.keys(emotionsMap).forEach(k => { emotionsMap[k] = emotionsMap[k] / totalSum; });

      const finalWordTimestamps = segs.map((seg, i) => ({
        word: seg.word || `${t("word_label")} ${i + 1}`,
        start: seg.start,
        end: seg.end,
        emotion: seg.emotion || "neutral",
        confidence: seg.confidence ?? 1.0,
      }));

      setAnalysisResult({
        dominant_emotion: emotion,
        emotions: emotionsMap,
        confidence: normalizedConfidence,
        word_timestamps: finalWordTimestamps,
        model_details: response.data.model_details || undefined,
      });
    } catch (error: any) {
      console.error("Analysis failed:", error);
      let msg = t("analysis_failed") || "Analiz hatası.";
      if (error.response?.data?.error) msg = `Hata: ${error.response.data.error}`;
      else if (error.message) msg = `Hata: ${error.message}`;
      alert(msg);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const reset = () => {
    setRecordedUrl(null);
    setAnalysisResult(null);
    setIsSegmenting(false);
    setIsAnalyzing(false);
  };

  return (
    <div className={clsx(
      "relative w-full flex-grow flex flex-col items-center font-sans transition-colors duration-500",
      isDark ? "bg-[#0b0f19] text-white" : "bg-gray-50 text-slate-900",
      analysisResult
        ? "justify-start pt-24 pb-12 overflow-x-hidden"
        : "justify-center"
    )}>
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <InteractiveBackground />
      </div>

      <div className="relative z-10 w-full max-w-6xl px-4 sm:px-6 flex flex-col items-center py-8 sm:py-12 md:py-16 mb-4 sm:mb-6 md:mb-8">

        {/* Başlık */}
        <h1 className={`font-outfit text-3xl sm:text-5xl md:text-7xl font-black mb-4 sm:mb-6 py-1 sm:py-2 leading-tight text-center tracking-tighter transition-all duration-700 ease-[cubic-bezier(0.16,1,0.3,1)] ${analysisResult ? "scale-75 mb-0 opacity-0 h-0" : "opacity-100"}`}>
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-fuchsia-500 to-rose-500 drop-shadow-sm">
            {t("discover_your_voice")}
          </span>
        </h1>

        {!analysisResult && (
          <p className={`text-base sm:text-xl md:text-2xl font-light mb-8 sm:mb-12 md:mb-16 text-center max-w-3xl leading-relaxed tracking-wide ${isDark ? "text-indigo-100/70" : "text-slate-600/90"} animate-slideUpFade`} style={{ animationDelay: "0.1s" }}>
            <Trans i18nKey="hero_subtitle">
              Gelişmiş <strong className="font-semibold text-indigo-400 dark:text-indigo-300">Üst Akıl</strong> algoritması ile sesinizdeki 4 temel duyguyu anında ve kusursuzca analiz edin.
            </Trans>
          </p>
        )}

        {/* Ana Kart */}
        <div className={clsx(
          "relative w-full backdrop-blur-[40px] shadow-2xl transition-all duration-1000 ease-[cubic-bezier(0.16,1,0.3,1)]",
          "flex flex-col items-center justify-center border",
          isDark ? "bg-[#0f172a]/70 border-white/10 shadow-[0_0_100px_rgba(99,102,241,0.15)]" : "bg-white/70 border-indigo-100/80 shadow-[0_0_100px_rgba(99,102,241,0.1)]",
          analysisResult
            ? "max-w-[100vw] sm:max-w-[98vw] lg:max-w-[1600px] min-h-[85vh] p-3 md:p-8 lg:p-10 overflow-visible rounded-3xl md:rounded-[2.5rem] mx-auto border-indigo-500/20"
            : "max-w-5xl min-h-[320px] p-6 md:p-10 rounded-[3rem]"
        )}
          style={{ marginTop: analysisResult ? "80px" : "0px" }}
        >
          {!analysisResult && (
            <div className="absolute inset-0 rounded-[3rem] pointer-events-none shadow-[inset_0_0_60px_rgba(255,255,255,0.05)]" />
          )}

          {/* Faz 1: Ses yok */}
          {!isSegmenting && !isAnalyzing && !analysisResult && (
            <div className="w-full flex-1 flex flex-col items-center justify-center animate-fadeIn z-10 relative">
              <AudioInput onAudioReady={handleAudioReady} />
            </div>
          )}

          {/* Faz 2: Segmentasyon spinner */}
          {isSegmenting && (
            <div className="flex flex-col items-center justify-center w-full h-full animate-fadeIn py-20 gap-4">
              <div className="relative w-20 h-20">
                <div className="absolute inset-0 border-4 border-indigo-200/20 rounded-full" />
                <div className="absolute inset-0 border-4 border-indigo-600 border-t-transparent rounded-full animate-[spin_1s_linear_infinite]" />
                <div className="absolute inset-2 border-4 border-purple-600 border-b-transparent rounded-full animate-[spin_1.5s_linear_infinite_reverse]" />
              </div>
              <p className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-500 to-purple-500 animate-pulse">
                {t("segmenting")}
              </p>
            </div>
          )}

          {/* Faz 3: Analiz spinner */}
          {isAnalyzing && (
            <div className="flex flex-col items-center justify-center w-full h-full animate-fadeIn py-20">
              <div className="relative w-24 h-24 mb-6">
                <div className="absolute inset-0 border-4 border-indigo-200/20 rounded-full" />
                <div className="absolute inset-0 border-4 border-indigo-600 border-t-transparent rounded-full animate-[spin_1s_linear_infinite]" />
                <div className="absolute inset-2 border-4 border-purple-600 border-b-transparent rounded-full animate-[spin_1.5s_linear_infinite_reverse]" />
              </div>
              <p className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-500 to-purple-500 animate-pulse">
                {t("analyzing")}
              </p>
              <p className="text-sm font-medium mt-2 tracking-widest uppercase opacity-50">
                {t("majority_voting")} · Studio
              </p>
            </div>
          )}

          {/* Faz 4: Sonuç */}
          {analysisResult && (
            <div className="w-full animate-fadeIn">
              <Result result={analysisResult} onBack={reset} audioUrl={recordedUrl || undefined} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MainHero;
