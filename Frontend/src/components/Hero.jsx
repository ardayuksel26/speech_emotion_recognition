import { useState, useRef, useEffect, useCallback } from "react";
import { useTranslation } from "react-i18next";
import { FaChevronLeft } from "react-icons/fa";
import { useTheme } from "../context/ThemeContext";
import ModeSelection from "./ModeSelection";
import FileUploader from "./FileUploader";
import AudioPlayer from "./AudioPlayer";
import Result from "./Result"; // YENİ: Result bileşeni eklendi
import axios from "axios";

// --- WAV ENCODER YARDIMCI FONKSİYONLARI ---
const writeString = (view, offset, string) => {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
};

const floatTo16BitPCM = (output, offset, input) => {
  for (let i = 0; i < input.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
};

const encodeWAV = (samples, sampleRate) => {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); 
  view.setUint16(22, 1, true); 
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true); 
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  floatTo16BitPCM(view, 44, samples);
  return view;
};

const convertFileToWav = async (file) => {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const ctx = new AudioCtx();
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
    const pcmData = audioBuffer.getChannelData(0);
    const wavDataView = encodeWAV(pcmData, audioBuffer.sampleRate);
    return new Blob([wavDataView], { type: "audio/wav" });
};

const Hero = () => {
  const { t } = useTranslation();
  const { isDark } = useTheme();
  const [mode, setMode] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  // State'ler
  const [recordingTime, setRecordingTime] = useState(0);
  const [recordedUrl, setRecordedUrl] = useState(null);
  const [savedLevels, setSavedLevels] = useState([]);
  const [audioFile, setAudioFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playProgress, setPlayProgress] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [isSpeedMenuOpen, setIsSpeedMenuOpen] = useState(false);

  // Ref'ler
  const streamRef = useRef(null);
  const audioElementRef = useRef(null);
  const playbackAnimationRef = useRef(null);
  const recordingIntervalRef = useRef(null);
  const isRecordingRef = useRef(false);
  
  // WAV Kayıt Ref'leri
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const audioInputRef = useRef(null);
  const audioDataRef = useRef([]); 
  
  // Görselleştirme Ref'leri
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);
  const timeDataRef = useRef(null);
  const allVisualizerDataRef = useRef([]); 
  const updateCounterRef = useRef(0);

  const BAR_COUNT = 60;

  // --- Yardımcılar ---
  const compressWaveform = (rawData, targetCount) => {
    if (!rawData || rawData.length === 0) return new Array(targetCount).fill(0.02);
    const blockSize = Math.floor(rawData.length / targetCount);
    const compressed = [];
    for (let i = 0; i < targetCount; i++) {
      const start = i * blockSize;
      let sum = 0;
      for (let j = 0; j < blockSize; j++) {
        if (rawData[start + j]) sum += rawData[start + j];
      }
      const avg = blockSize > 0 ? sum / blockSize : 0;
      compressed.push(Math.max(0.05, avg));
    }
    return compressed;
  };

  const processUploadedFile = async (file) => {
    if (!file) return;
    setAudioFile(file);
    const url = URL.createObjectURL(file);
    setRecordedUrl(url);

    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioCtx();

    try {
      const arrayBuffer = await file.arrayBuffer();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      setRecordingTime(audioBuffer.duration);

      const rawData = audioBuffer.getChannelData(0);
      const blockSize = Math.floor(rawData.length / BAR_COUNT);
      const samples = [];
      for (let i = 0; i < BAR_COUNT; i++) {
        let sum = 0;
        for (let j = 0; j < blockSize; j++) {
          const val = rawData[i * blockSize + j];
          sum += val * val;
        }
        samples.push(Math.sqrt(sum / blockSize));
      }
      const maxVal = Math.max(...samples, 0.01);
      const normalized = samples.map(s => Math.max(0.05, Math.min(1, Math.pow(s / maxVal, 0.8))));
      setSavedLevels(normalized);
    } catch (err) {
      console.error(err);
      alert(t('audio_processing_failed'));
    }
  };

  const startVisualizer = () => {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const draw = () => {
      animationFrameRef.current = requestAnimationFrame(draw);
      const bufferLength = analyser.fftSize;
      if (!timeDataRef.current || timeDataRef.current.length !== bufferLength) {
        timeDataRef.current = new Uint8Array(bufferLength);
      }
      analyser.getByteTimeDomainData(timeDataRef.current);

      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        const v = (timeDataRef.current[i] - 128) / 128;
        sum += v * v;
      }
      let rms = Math.sqrt(sum / bufferLength);
      rms = Math.pow(rms, 0.6) * 2.5;

      allVisualizerDataRef.current.push(rms);
      updateCounterRef.current++;
      
      if (updateCounterRef.current % 5 === 0) {
        setSavedLevels(prev => {
           const view = allVisualizerDataRef.current.slice(-BAR_COUNT);
           while(view.length < BAR_COUNT) view.unshift(0.02);
           return view;
        });
      }
    };
    draw();
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      setRecordingTime(0);
      setRecordedUrl(null);
      setAudioFile(null);
      setAnalysisResult(null);
      setSavedLevels(new Array(BAR_COUNT).fill(0.02));
      audioDataRef.current = [];
      allVisualizerDataRef.current = [];

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const audioContext = new AudioCtx();
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      audioInputRef.current = source;

      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;

      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (!isRecordingRef.current) return;
        const channelData = e.inputBuffer.getChannelData(0);
        audioDataRef.current.push(new Float32Array(channelData));
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      startVisualizer();
      setIsRecording(true);
      isRecordingRef.current = true;

      recordingIntervalRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

    } catch (err) {
      console.error(err);
      alert(t('mic_error'));
    }
  };

  const stopRecording = () => {
    if (!isRecording) return;
    
    // Önce recording'i durdur
    setIsRecording(false);
    isRecordingRef.current = false;
    clearInterval(recordingIntervalRef.current);
    
    // Animation'ı durdur
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    
    // Stream'i durdur
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    
    // Audio processor'ı temizle
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current.onaudioprocess = null;
      processorRef.current = null;
    }
    
    if (audioInputRef.current) {
      audioInputRef.current.disconnect();
      audioInputRef.current = null;
    }

    // Kaydedilen veriyi işle
    if (audioDataRef.current.length > 0 && audioContextRef.current) {
      try {
        const sampleRate = audioContextRef.current.sampleRate;
        const totalLength = audioDataRef.current.reduce((acc, buf) => acc + buf.length, 0);
        const result = new Float32Array(totalLength);
        
        let offset = 0;
        for (const buf of audioDataRef.current) {
          result.set(buf, offset);
          offset += buf.length;
        }

        const wavData = encodeWAV(result, sampleRate);
        const blob = new Blob([wavData], { type: "audio/wav" });
        
        setAudioFile(blob);
        const url = URL.createObjectURL(blob);
        setRecordedUrl(url);

        const fullWaveform = compressWaveform(allVisualizerDataRef.current, BAR_COUNT);
        setSavedLevels(fullWaveform);
      } catch (error) {
        console.error("Error processing recording:", error);
      }
    }

    // Audio context'i kapat
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
  };

  const resetRecording = () => {
    setRecordedUrl(null);
    setAudioFile(null);
    setAnalysisResult(null);
    setRecordingTime(0);
    setSavedLevels([]);
    setPlayProgress(0);
    setIsPlaying(false);
    setPlaybackRate(1.0);
    setIsSpeedMenuOpen(false);
  };

  const togglePlayPause = () => {
    if (!audioElementRef.current || !recordedUrl) return;
    if (isPlaying) {
      audioElementRef.current.pause();
      setIsPlaying(false);
      if (playbackAnimationRef.current) cancelAnimationFrame(playbackAnimationRef.current);
    } else {
      if (audioElementRef.current.ended || audioElementRef.current.currentTime === 0) {
        audioElementRef.current.currentTime = 0;
      }
      audioElementRef.current.play();
      audioElementRef.current.playbackRate = playbackRate;
      setIsPlaying(true);
    }
  };

  const updatePlayProgressSmoothly = useCallback(() => {
    if (audioElementRef.current && !audioElementRef.current.paused && !audioElementRef.current.ended) {
      const duration = audioElementRef.current.duration;
      const currentTime = audioElementRef.current.currentTime;
      if (duration > 0) setPlayProgress(currentTime / duration);
      playbackAnimationRef.current = requestAnimationFrame(updatePlayProgressSmoothly);
    }
  }, []);

  useEffect(() => {
    if (isPlaying) playbackAnimationRef.current = requestAnimationFrame(updatePlayProgressSmoothly);
    else if (playbackAnimationRef.current) cancelAnimationFrame(playbackAnimationRef.current);
    return () => { if (playbackAnimationRef.current) cancelAnimationFrame(playbackAnimationRef.current); };
  }, [isPlaying, updatePlayProgressSmoothly]);

  const handleSpeedChange = (rate) => {
    setPlaybackRate(rate);
    if (audioElementRef.current) audioElementRef.current.playbackRate = rate;
  };

  const goBack = () => {
    setMode(null);
    setRecordedUrl(null);
    setAudioFile(null);
    setAnalysisResult(null);
    setSavedLevels([]);
    setRecordingTime(0);
    setIsRecording(false);
    setPlayProgress(0);
    setIsPlaying(false);
    setIsSpeedMenuOpen(false);
  };

  const handleAnalyze = async () => {
    if (!audioFile) {
        alert(t('no_audio_file'));
        return;
    }
    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
        const wavBlob = await convertFileToWav(audioFile);
        const formData = new FormData();
        formData.append("file", wavBlob, "converted_audio.wav");

        const response = await axios.post("http://localhost:8000/predict", formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        setAnalysisResult(response.data);
        
    } catch (error) {
        console.error("Analiz Hatası:", error);
        let msg = t('analysis_failed') || "Analiz hatası.";
        if (error.response && error.response.data && error.response.data.detail) {
            msg += ` (${error.response.data.detail})`;
        }
        alert(msg);
    } finally {
        setIsAnalyzing(false);
    }
  };

  useEffect(() => {
    return () => {
      clearInterval(recordingIntervalRef.current);
      if (playbackAnimationRef.current) cancelAnimationFrame(playbackAnimationRef.current);
    };
  }, []);

  return (
    <div className={`relative w-full flex flex-col items-center justify-center overflow-hidden font-sans transition-colors duration-300 ${isDark ? "bg-slate-900 selection:bg-indigo-500 selection:text-white" : "bg-gray-100 selection:bg-blue-500 selection:text-white"}`}>
      <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-indigo-600 rounded-full mix-blend-multiply filter blur-[128px] opacity-40 animate-blob"></div>
      <div className="absolute top-[-10%] right-[-10%] w-96 h-96 bg-purple-600 rounded-full mix-blend-multiply filter blur-[128px] opacity-40 animate-blob animation-delay-2000"></div>
      <div className="absolute -bottom-32 left-20 w-96 h-96 bg-pink-600 rounded-full mix-blend-multiply filter blur-[128px] opacity-40 animate-blob animation-delay-4000"></div>

      <div className="relative z-10 w-full max-w-5xl px-6 flex flex-col items-center">
        <div className="relative text-center mb-10 -top-5">
          <h1 className={`py-6 pb-8 text-4xl md:text-6xl font-extrabold tracking-tight overflow-visible ${isDark ? "text-transparent bg-clip-text bg-gradient-to-r from-white via-indigo-100 to-indigo-200" : "text-gray-800"}`}>
            {t('discover_your_voice')}
          </h1>
        </div>

        <div className={`w-full backdrop-blur-xl rounded-lg shadow-2xl transition-all duration-500 min-h-[400px] flex flex-col relative ${isDark ? "bg-slate-800/40 border border-white/10" : "bg-white/80 border border-gray-200"} ${isSpeedMenuOpen ? 'overflow-visible' : 'overflow-hidden'}`}>
          
          <div className="flex-1 p-8 flex flex-col items-center justify-center h-full">
            {!mode && !analysisResult && <ModeSelection onSelectMode={setMode} />}
            
            {mode === "upload" && !recordedUrl && !analysisResult && (
              <FileUploader onFileSelect={processUploadedFile} isDragging={isDragging} setIsDragging={setIsDragging} />
            )}
            
            {mode === "record" && !analysisResult && (
              <AudioPlayer mode={mode} isRecording={isRecording} recordedUrl={recordedUrl} recordingTime={recordingTime} levels={savedLevels} isPlaying={isPlaying} playProgress={playProgress} playbackRate={playbackRate} onStartRecording={startRecording} onStopRecording={stopRecording} onTogglePlay={togglePlayPause} onSpeedChange={handleSpeedChange} onAnalyze={handleAnalyze} onBack={goBack} isSpeedMenuOpen={isSpeedMenuOpen} setIsSpeedMenuOpen={setIsSpeedMenuOpen} isLoading={isAnalyzing} />
            )}
            
            {mode === "upload" && recordedUrl && !analysisResult && (
              <AudioPlayer mode={mode} isRecording={isRecording} recordedUrl={recordedUrl} recordingTime={recordingTime} levels={savedLevels} isPlaying={isPlaying} playProgress={playProgress} playbackRate={playbackRate} onStartRecording={startRecording} onStopRecording={stopRecording} onTogglePlay={togglePlayPause} onSpeedChange={handleSpeedChange} onAnalyze={handleAnalyze} onBack={goBack} isSpeedMenuOpen={isSpeedMenuOpen} setIsSpeedMenuOpen={setIsSpeedMenuOpen} isLoading={isAnalyzing} />
            )}
            
            {/* Yükleniyor Göstergesi */}
            {isAnalyzing && (
                <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-black/50 backdrop-blur-sm rounded-3xl">
                    <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-indigo-500 mb-4"></div>
                    <p className="text-white font-semibold text-lg animate-pulse">{t('analyzing') || "Yapay Zeka Analiz Ediyor..."}</p>
                </div>
            )}

            {/* YENİ RESULT BİLEŞENİ ENTEGRASYONU */}
            {analysisResult && !isAnalyzing && (
                <div className={`absolute inset-0 z-40 transition-all duration-300 rounded-lg ${isDark ? "bg-slate-800" : "bg-white"}`}>
                   <Result 
                     result={analysisResult} 
                     onBack={goBack}
                     audioUrl={recordedUrl}
                     waveformLevels={savedLevels}
                   />
                </div>
            )}

            {recordedUrl && !analysisResult && (
              <audio ref={audioElementRef} src={recordedUrl} onEnded={() => { setIsPlaying(false); setPlayProgress(0); if (playbackAnimationRef.current) cancelAnimationFrame(playbackAnimationRef.current); if (audioElementRef.current) audioElementRef.current.currentTime = 0; }} className="hidden" />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero;