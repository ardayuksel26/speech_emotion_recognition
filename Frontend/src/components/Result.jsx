import React, { useState, useRef, useEffect, useCallback } from "react";
import { useTranslation } from "react-i18next";
import { FaChevronLeft, FaChartBar, FaPlay, FaPause } from "react-icons/fa";
import { MdSpeed } from "react-icons/md";
import { useTheme } from "../context/ThemeContext";

const Result = ({ result, onBack, audioUrl, waveformLevels = [] }) => {
  const { t } = useTranslation();
  const { isDark } = useTheme();
  const [activeTab, setActiveTab] = useState("results"); // "audio" veya "results"
  const [isPlaying, setIsPlaying] = useState(false);
  const [playProgress, setPlayProgress] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [isSpeedMenuOpen, setIsSpeedMenuOpen] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  
  const audioRef = useRef(null);
  const speedMenuRef = useRef(null);
  const playbackAnimationRef = useRef(null);
  
  const speedOptions = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];

  // Backend'den gelen veri yapısı: { prediction, confidence, all_probabilities, details }
  const { prediction, confidence, all_probabilities, details } = result;

  const translateEmotion = (emotion) => {
    const map = {
      angry: t('emotion_angry') || "Kızgın",
      calm: t('emotion_calm') || "Sakin",
      happy: t('emotion_happy') || "Mutlu",
      sad: t('emotion_sad') || "Üzgün"
    };
    return map[emotion] || emotion;
  };

  const getEmotionEmoji = (emotion) => {
    const map = { angry: "😡", calm: "😌", happy: "😊", sad: "😢" };
    return map[emotion] || "😐";
  };

  const getEmotionColor = (emotion) => {
    const map = {
      angry: "from-red-500 to-orange-500",
      calm: "from-blue-400 to-teal-400",
      happy: "from-yellow-400 to-orange-400",
      sad: "from-indigo-400 to-purple-500"
    };
    return map[emotion] || "from-gray-400 to-gray-500";
  };

  const getEmotionBg = (emotion) => {
    const map = {
      angry: "bg-red-500",
      calm: "bg-blue-400",
      happy: "bg-yellow-400",
      sad: "bg-indigo-400"
    };
    return map[emotion] || "bg-gray-400";
  };

  // Olasılıkları büyükten küçüğe sırala
  const sortedProbs = Object.entries(all_probabilities || {})
    .sort(([, a], [, b]) => b - a);

  const getComment = (emotion) => {
    const map = {
        angry: t('comment_angry') || "Ses tonunuzda yüksek enerji ve gerginlik tespit edildi. Biraz mola vermek iyi olabilir.",
        calm: t('comment_calm') || "Sesiniz gayet dengeli ve huzurlu geliyor. Bu sakinliği koruyun!",
        happy: t('comment_happy') || "Harika bir enerji! Sesinizdeki pozitiflik etrafa neşe saçıyor.",
        sad: t('comment_sad') || "Ses tonunuzda biraz hüzün var. Kendinize vakit ayırmayı unutmayın."
    };
    return map[emotion] || "";
  };

  const formatTime = (seconds) => {
    const m = String(Math.floor(seconds / 60)).padStart(2, "0");
    const s = String(Math.floor(seconds % 60)).padStart(2, "0");
    return `${m}:${s}`;
  };

  const togglePlayPause = () => {
    if (!audioRef.current || !audioUrl) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
      if (playbackAnimationRef.current) cancelAnimationFrame(playbackAnimationRef.current);
    } else {
      if (audioRef.current.ended || audioRef.current.currentTime === 0) {
        audioRef.current.currentTime = 0;
      }
      audioRef.current.play();
      audioRef.current.playbackRate = playbackRate;
      setIsPlaying(true);
    }
  };

  const handleSpeedChange = (rate) => {
    setPlaybackRate(rate);
    if (audioRef.current) audioRef.current.playbackRate = rate;
    setIsSpeedMenuOpen(false);
  };

  const updatePlayProgressSmoothly = useCallback(() => {
    if (audioRef.current && !audioRef.current.paused && !audioRef.current.ended) {
      const duration = audioRef.current.duration;
      const currentTime = audioRef.current.currentTime;
      if (duration > 0) {
        setPlayProgress(currentTime / duration);
        setRecordingTime(currentTime);
      }
      playbackAnimationRef.current = requestAnimationFrame(updatePlayProgressSmoothly);
    }
  }, []);

  useEffect(() => {
    if (isPlaying) {
      playbackAnimationRef.current = requestAnimationFrame(updatePlayProgressSmoothly);
    } else if (playbackAnimationRef.current) {
      cancelAnimationFrame(playbackAnimationRef.current);
    }
    return () => {
      if (playbackAnimationRef.current) cancelAnimationFrame(playbackAnimationRef.current);
    };
  }, [isPlaying, updatePlayProgressSmoothly]);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (speedMenuRef.current && !speedMenuRef.current.contains(event.target)) {
        setIsSpeedMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    if (audioRef.current && audioUrl) {
      const duration = audioRef.current.duration;
      if (!isNaN(duration)) {
        setRecordingTime(duration);
      }
    }
  }, [audioUrl]);

  return (
    <div className="w-full h-full flex flex-col animate-fadeIn">
      {/* Üst Bar */}
      <div className="flex items-center justify-between mb-4 flex-shrink-0 pt-2">
        <button 
          onClick={onBack}
          className={`p-3 rounded-full transition-colors flex items-center justify-center ${
            isDark ? "hover:bg-white/10 text-gray-300" : "hover:bg-gray-100 text-gray-600"
          }`}
        >
          <FaChevronLeft className="text-lg" />
        </button>
        
        {/* Sekme Butonları */}
        <div className={`flex gap-4 rounded-xl p-2 ${isDark ? "bg-slate-700/50" : "bg-gray-200"}`}>
          <button
            onClick={() => setActiveTab("audio")}
            className={`min-w-[140px] px-8 py-4 rounded-lg font-bold text-lg transition-all ${
              activeTab === "audio"
                ? isDark
                  ? "bg-indigo-600 text-white shadow-lg"
                  : "bg-white text-gray-900 shadow-lg"
                : isDark
                ? "text-gray-400 hover:text-gray-200"
                : "text-gray-600 hover:text-gray-900"
            }`}
          >
            {t('audio') || "Ses"}
          </button>
          <button
            onClick={() => setActiveTab("results")}
            className={`min-w-[140px] px-8 py-4 rounded-lg font-bold text-lg transition-all ${
              activeTab === "results"
                ? isDark
                  ? "bg-indigo-600 text-white shadow-lg"
                  : "bg-white text-gray-900 shadow-lg"
                : isDark
                ? "text-gray-400 hover:text-gray-200"
                : "text-gray-600 hover:text-gray-900"
            }`}
          >
            {t('results') || "Sonuçlar"}
          </button>
        </div>
        
        <div className="w-9" /> {/* Dengeleyici boşluk */}
      </div>

      <div className="flex-1 overflow-y-auto overflow-x-hidden custom-scrollbar px-2 pb-4">
        
        {/* SES SEKMESİ */}
        {activeTab === "audio" && audioUrl && (
          <div className="w-full flex flex-col items-center animate-fadeIn">
            
            {/* Waveform Container - AudioPlayer ile aynı */}
            <div className={`w-full h-40 rounded-2xl border p-4 mb-4 relative overflow-hidden flex items-center justify-center ${
              isDark 
                ? "bg-slate-800/60 border-white/5" 
                : "bg-purple-100 border-purple-200"
            }`}>
              <div className="flex items-center justify-center gap-1 w-full h-full">
                {waveformLevels.map((level, index) => {
                  const totalBars = waveformLevels.length;
                  const playIndex = Math.floor(playProgress * (totalBars - 1));
                  const isCurrent = index === playIndex && isPlaying;
                  const isPassed = index < playIndex;

                  return (
                    <div
                      key={index}
                      className={`flex-1 rounded-full transition-all duration-75 ease-out
                      ${isCurrent 
                        ? isDark 
                          ? 'bg-white shadow-[0_0_15px_rgba(255,255,255,0.8)] scale-110'
                          : 'bg-purple-700 shadow-[0_0_15px_rgba(147,51,234,0.8)] scale-110'
                        : isPassed 
                        ? isDark 
                          ? 'bg-indigo-500/40'
                          : 'bg-purple-400'
                        : isDark
                        ? 'bg-gradient-to-t from-indigo-600 to-purple-500'
                        : 'bg-gradient-to-t from-purple-500 to-indigo-500'
                      }`}
                      style={{
                        height: `${Math.max(4, level * 100)}px`,
                        opacity: isCurrent ? 1 : isPassed ? 0.6 : 0.4 + level * 0.6,
                        transform: isCurrent ? 'scaleY(1.1)' : 'scaleY(1)', 
                      }}
                    />
                  );
                })}
              </div>
            </div>

            {/* Zaman Göstergesi */}
            <div className={`font-mono text-4xl font-bold mb-6 tracking-widest drop-shadow-lg ${
              isDark ? "text-white" : "text-gray-800"
            }`}>
              {formatTime(recordingTime)}
            </div>

            {/* Kontrol Butonları */}
            <div className="flex items-center gap-6 md:gap-6 relative z-10">
              {/* Play/Pause Butonu */}
              <button 
                onClick={togglePlayPause} 
                className="w-20 h-20 rounded-full bg-indigo-500 hover:bg-indigo-400 flex items-center justify-center text-white shadow-[0_0_30px_rgba(99,102,241,0.5)] hover:scale-105 transition-all"
              >
                {isPlaying ? <FaPause className="text-2xl" /> : <FaPlay className="text-2xl ml-1" />}
              </button>

              {/* Hız Ayarlama Butonu */}
              <div className="relative" ref={speedMenuRef}>
                <button 
                  onClick={() => setIsSpeedMenuOpen(!isSpeedMenuOpen)} 
                  className={`w-12 h-12 rounded-full flex items-center justify-center transition-all cursor-pointer ${
                    isDark
                      ? "bg-indigo-600/80 hover:bg-indigo-500 border border-indigo-400/30"
                      : "bg-indigo-500 hover:bg-indigo-400 border border-indigo-300"
                  }`}
                >
                  {playbackRate === 1.0 ? (
                    <MdSpeed className="text-lg text-white" />
                  ) : (
                    <span className="text-xs font-bold text-white">{playbackRate}x</span>
                  )}
                </button>

                {isSpeedMenuOpen && (
                  <div className={`absolute bottom-full mb-3 left-1/2 -translate-x-1/2 w-32 rounded-xl shadow-2xl overflow-hidden py-1 z-[100] ${
                    isDark ? "bg-slate-800 border border-slate-700" : "bg-white border border-gray-200"
                  }`}>
                    {speedOptions.map((rate) => (
                      <button
                        key={rate}
                        onClick={() => handleSpeedChange(rate)}
                        className={`block w-full text-center px-4 py-2.5 text-sm font-medium transition-colors duration-150
                          ${rate === playbackRate 
                            ? 'bg-indigo-600 text-white' 
                            : isDark 
                              ? 'text-white hover:bg-slate-700' 
                              : 'text-gray-700 hover:bg-gray-100'
                          }`}
                      >
                        {rate}x {t('speed') || 'Hız'}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
            
            {/* Gizli Audio Element */}
            <audio 
              ref={audioRef}
              src={audioUrl}
              onEnded={() => { 
                setIsPlaying(false); 
                setPlayProgress(0); 
                if (playbackAnimationRef.current) cancelAnimationFrame(playbackAnimationRef.current);
                if (audioRef.current) audioRef.current.currentTime = 0;
              }}
              onLoadedMetadata={() => {
                if (audioRef.current) {
                  setRecordingTime(audioRef.current.duration);
                }
              }}
              className="hidden"
            />
          </div>
        )}
        
        {/* SONUÇLAR SEKMESİ */}
        {activeTab === "results" && (
          <div className="animate-fadeIn">
            {/* Ana Sonuç Kartı */}
        <div className={`relative overflow-hidden rounded-2xl p-8 mb-6 text-center shadow-2xl ${
           isDark ? "bg-gradient-to-br from-slate-800 to-slate-900" : "bg-gradient-to-br from-white to-gray-50"
        }`}>
          
          {/* Arka plan efektleri */}
          <div className={`absolute -top-20 -right-20 w-40 h-40 rounded-full opacity-20 blur-3xl ${getEmotionBg(prediction)}`} />
          <div className={`absolute -bottom-20 -left-20 w-40 h-40 rounded-full opacity-20 blur-3xl ${getEmotionBg(prediction)}`} />

          <div className="relative z-10 flex flex-col items-center">
            {/* Emoji */}
            <div className={`w-28 h-28 rounded-full flex items-center justify-center text-6xl mb-6 shadow-2xl bg-gradient-to-br ${getEmotionColor(prediction)} transform hover:scale-110 transition-transform duration-300`}>
              {getEmotionEmoji(prediction)}
            </div>
            
            {/* Duygu Adı */}
            <h1 className={`text-4xl font-black mb-6 bg-gradient-to-r ${getEmotionColor(prediction)} bg-clip-text text-transparent`}>
              {translateEmotion(prediction)}
            </h1>

            {/* Güven Skoru */}
            <div className="w-full max-w-md">
              <div className="flex justify-between items-center mb-3">
                <span className={`text-sm font-bold ${isDark ? "text-gray-300" : "text-gray-700"}`}>
                  {t('confidence_score') || "Güven Skoru"}
                </span>
                <span className={`text-2xl font-black bg-gradient-to-r ${getEmotionColor(prediction)} bg-clip-text text-transparent`}>
                  %{Math.round(confidence * 100)}
                </span>
              </div>
              <div className={`h-3 rounded-full overflow-hidden shadow-inner ${isDark ? "bg-slate-700" : "bg-gray-200"}`}>
                <div 
                  className={`h-full rounded-full bg-gradient-to-r ${getEmotionColor(prediction)} shadow-lg relative overflow-hidden`}
                  style={{ width: `${confidence * 100}%`, transition: "width 1.5s cubic-bezier(0.4, 0, 0.2, 1)" }}
                >
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer"></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Detaylı İstatistikler */}
        <div className={`rounded-2xl p-5 shadow-xl ${
          isDark ? "bg-slate-800/50" : "bg-white"
        }`}>
          <div className="flex items-center gap-2 mb-4">
             <FaChartBar className={isDark ? "text-indigo-400" : "text-indigo-600"} />
             <h3 className={`font-bold ${isDark ? "text-gray-200" : "text-gray-700"}`}>
               {t('probability_breakdown') || "Olasılık Dağılımı"}
             </h3>
          </div>

          <div className="space-y-3">
            {sortedProbs.map(([emotion, score]) => (
                <div key={emotion} className={`p-3 rounded-xl flex items-center gap-4 ${
                isDark ? "bg-white/5 border border-white/5" : "bg-gray-50 border border-gray-100"
                }`}>
                <div className={`w-10 h-10 rounded-full flex items-center justify-center text-xl bg-gradient-to-br ${getEmotionColor(emotion)} bg-opacity-20`}>
                    {getEmotionEmoji(emotion)}
                </div>
                
                <div className="flex-1">
                    <div className="flex justify-between mb-1 items-end">
                    <span className={`font-semibold capitalize ${isDark ? "text-gray-200" : "text-gray-700"}`}>
                        {translateEmotion(emotion)}
                    </span>
                    <span className={`text-sm font-mono ${isDark ? "text-gray-400" : "text-gray-500"}`}>
                        %{(score * 100).toFixed(1)}
                    </span>
                    </div>
                    <div className={`h-2 rounded-full overflow-hidden ${isDark ? "bg-black/20" : "bg-gray-200"}`}>
                    <div 
                        className={`h-full rounded-full bg-gradient-to-r ${getEmotionColor(emotion)} opacity-90`}
                        style={{ width: `${score * 100}%` }}
                    />
                    </div>
                </div>
                </div>
            ))}
          </div>
        </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Result;