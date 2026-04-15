import { FaMicrophone, FaPlay, FaPause, FaMagic, FaChevronLeft } from "react-icons/fa";
import { MdSpeed } from "react-icons/md";
import { useRef, useEffect } from "react";
import { useTheme } from "../context/ThemeContext";
import { useTranslation } from "react-i18next";

const AudioPlayer = ({
  mode, // 'record' or 'preview'
  analysisMode, // 'word' or 'sentence' (passed from parent)
  selectedModelName, // The dynamically selected model to show in the action button
  isRecording,
  recordedUrl,
  recordingTime,
  levels,
  isPlaying,
  playProgress,
  playbackRate,
  onStartRecording,
  onStopRecording,
  onTogglePlay,
  onSpeedChange,
  onAnalyze,
  onBack,
  isSpeedMenuOpen,
  setIsSpeedMenuOpen,
  duration,
  currentTime,
  showAnalyzeButton = true,
}) => {
  const { isDark } = useTheme();
  const { t } = useTranslation();
  const speedMenuRef = useRef(null);
  const speedOptions = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];

  const formatTime = (seconds) => {
    const m = String(Math.floor(seconds / 60)).padStart(2, "0");
    const s = String(Math.floor(seconds % 60)).padStart(2, "0");
    return `${m}:${s}`;
  };

  // Decide button text based on analysis mode
  const getAnalyzeButtonText = () => {
    if (analysisMode === 'sentence') {
      return `${selectedModelName || 'CatBoost'} ${t('sentence_analysis')}`;
    }
    return t('analyze_audio') || 'Ses Analizini Yap';
  };

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (speedMenuRef.current && !speedMenuRef.current.contains(event.target)) {
        setIsSpeedMenuOpen(false); // Prop olan fonksiyonu çağırır
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [setIsSpeedMenuOpen]); // Dependency array güncellendi

  return (
    <div className="w-full flex flex-col items-center animate-fade-in-up relative">

      {/* Geri Butonu - Sadece kayıt yapıldıysa göster */}
      {recordedUrl && !isRecording && (
        <button
          onClick={onBack}
          className={`absolute top-2 left-2 w-10 h-10 rounded-full flex items-center justify-center transition-all duration-200 z-30 shadow-sm hover:shadow-md hover:scale-105 ${isDark ? "bg-slate-700/50 text-slate-200 hover:bg-slate-600" : "bg-white/80 text-slate-600 hover:bg-white"
            }`}
        >
          <FaChevronLeft className="text-sm" />
        </button>
      )}

      <div className={`w-full h-40 rounded-2xl border p-4 mb-4 relative overflow-hidden flex items-center justify-center ${isDark
        ? "bg-slate-800/60 border-white/5"
        : "bg-purple-100 border-purple-200"
        }`}>
        {!isRecording && !recordedUrl ? (
          <div className={`flex flex-col items-center justify-center h-full animate-pulse ${isDark ? "text-slate-500" : "text-purple-600"
            }`}>
            <div className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 ${isDark ? "bg-white/5" : "bg-purple-200"
              }`}>
              <FaMicrophone className={`text-lg ${isDark ? "text-slate-400" : "text-purple-600"
                }`} />
            </div>
            <span className="text-xs tracking-wider uppercase font-semibold">{t('click_to_record')}</span>
          </div>
        ) : (
          <div className="flex items-center justify-center gap-1 w-full h-full">
            {levels.map((level, index) => {
              const totalBars = levels.length;
              // Use continuous float index for smooth progress
              const floatIndex = playProgress * (totalBars - 1);
              const isPassed = index < floatIndex - 0.5;
              const isCurrent = !isPassed && index < floatIndex + 0.5;

              // Smooth glow intensity for the current bar (0→1→0 as it passes)
              const distFromHead = Math.abs(index - floatIndex);
              const glowStrength = Math.max(0, 1 - distFromHead);

              let background;
              if (isDark) {
                background = isPassed
                  ? 'rgba(99,102,241,0.35)'
                  : `linear-gradient(to top, #4f46e5, #8b5cf6)`;
              } else {
                background = isPassed
                  ? '#c4b5fd'
                  : `linear-gradient(to top, #7c3aed, #6366f1)`;
              }

              const boxShadow = glowStrength > 0.1
                ? isDark
                  ? `0 0 ${12 * glowStrength}px rgba(255,255,255,${0.7 * glowStrength})`
                  : `0 0 ${12 * glowStrength}px rgba(139,92,246,${0.8 * glowStrength})`
                : 'none';

              return (
                <div
                  key={index}
                  className="flex-1 rounded-full"
                  style={{
                    height: `${Math.max(4, level * 100)}px`,
                    background,
                    boxShadow,
                    opacity: isPassed ? 0.55 : 0.4 + level * 0.6,
                    transform: glowStrength > 0.4 ? `scaleY(${1 + 0.1 * glowStrength})` : 'scaleY(1)',
                    transition: 'opacity 0.05s linear, box-shadow 0.05s linear, transform 0.05s linear',
                  }}
                />
              );
            })}
          </div>
        )}
      </div>

      <div className={`font-mono text-4xl font-bold mb-6 tracking-widest drop-shadow-lg ${isDark ? "text-white" : "text-gray-800"
        }`}>
        {recordedUrl && duration > 0
          ? `${formatTime(currentTime || 0)} / ${formatTime(duration)}`
          : formatTime(recordingTime)
        }
      </div>

      <div className="flex items-center gap-6 md:gap-6 relative z-10 mb-6">

        {mode === "record" && !isRecording && !recordedUrl && (
          <button onClick={onStartRecording} className="group relative w-20 h-20 rounded-full flex items-center justify-center bg-white hover:scale-105 transition-all duration-300 shadow-[0_0_40px_rgba(255,255,255,0.3)]">
            <div className="absolute inset-0 rounded-full border border-slate-200 animate-ping opacity-20"></div>
            <FaMicrophone className="text-3xl text-indigo-600 group-hover:text-indigo-800 transition-colors" />
          </button>
        )}

        {isRecording && (
          <button onClick={onStopRecording} className="w-20 h-20 rounded-full flex items-center justify-center bg-red-500 hover:bg-red-600 hover:scale-105 transition-all duration-300 shadow-[0_0_30px_rgba(239,68,68,0.4)]">
            <div className="w-8 h-8 bg-white rounded-md"></div>
          </button>
        )}

        {!isRecording && recordedUrl && (
          <>
            <button onClick={onTogglePlay} className="w-20 h-20 rounded-full bg-indigo-500 hover:bg-indigo-400 flex items-center justify-center text-white shadow-[0_0_30px_rgba(99,102,241,0.5)] hover:scale-105 transition-all">
              {isPlaying ? <FaPause className="text-2xl" /> : <FaPlay className="text-2xl ml-1" />}
            </button>

            <div className="relative" ref={speedMenuRef}>
              <button
                onClick={() => setIsSpeedMenuOpen(!isSpeedMenuOpen)}
                className={`w-12 h-12 rounded-full flex items-center justify-center transition-all cursor-pointer ${isDark
                  ? "bg-indigo-600/80 hover:bg-indigo-500 border border-indigo-400/30"
                  : "bg-indigo-500 hover:bg-indigo-400 border border-indigo-300"
                  }`}
              >
                {playbackRate === 1.0 ? (
                  <MdSpeed className={`text-lg ${isDark ? "text-white" : "text-white"}`} />
                ) : (
                  <span className="text-xs font-bold text-white">{playbackRate}x</span>
                )}
              </button>

              {isSpeedMenuOpen && (
                <div
                  className="absolute bottom-full mb-3 left-1/2 -translate-x-1/2 w-32 rounded-sm shadow-2xl overflow-hidden py-1 z-[100]"
                  style={{
                    background: isDark ? 'rgba(5, 5, 5, 0.98)' : '#ffffff',
                    border: isDark ? '1px solid rgba(255,255,255,0.1)' : '1px solid #e2e8f0',
                  }}
                >
                  {speedOptions.map((rate) => (
                    <button
                      key={rate}
                      onClick={() => { onSpeedChange(rate); setIsSpeedMenuOpen(false); }}
                      className="block w-full text-center px-4 py-2.5 text-sm font-medium transition-colors duration-150"
                      style={{
                        background: rate === playbackRate ? '#4f46e5' : 'transparent',
                        color: rate === playbackRate ? '#ffffff' : (isDark ? '#e2e8f0' : '#1e293b'),
                      }}
                      onMouseEnter={e => {
                        if (rate !== playbackRate) e.currentTarget.style.background = isDark ? '#334155' : '#f1f5f9';
                      }}
                      onMouseLeave={e => {
                        if (rate !== playbackRate) e.currentTarget.style.background = 'transparent';
                      }}
                    >
                      {rate}x {t('speed') || 'Hız'}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </>
        )}
      </div>

      {/* Analyze Button - Kontrol butonlarının altında */}
      {!isRecording && recordedUrl && showAnalyzeButton && (
        <button
          onClick={onAnalyze}
          className="transition-all duration-300 rounded-xl text-base font-bold hover:scale-[1.02]"
          style={{
            marginTop: '24px',
            padding: '14px 48px',
            background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
            color: 'white',
            boxShadow: '0 4px 15px rgba(99,102,241,0.3)',
          }}
        >
          {getAnalyzeButtonText()}
        </button>
      )}
    </div>
  );
};

export default AudioPlayer;