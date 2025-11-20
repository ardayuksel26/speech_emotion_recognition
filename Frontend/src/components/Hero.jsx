import { useState, useRef, useEffect, useCallback } from "react";
import { FaChevronLeft } from "react-icons/fa";
import ModeSelection from "./ModeSelection";
import FileUploader from "./FileUploader";
import AudioPlayer from "./AudioPlayer";

const Hero = () => {
  const [mode, setMode] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  // Ortak State'ler
  const [recordingTime, setRecordingTime] = useState(0);
  const [recordedUrl, setRecordedUrl] = useState(null);
  const [savedLevels, setSavedLevels] = useState([]); 
  
  // Kayıt State'leri
  const [isRecording, setIsRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);

  // Oynatma State'leri
  const [isPlaying, setIsPlaying] = useState(false);
  const [playProgress, setPlayProgress] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  
  // Hız menüsü state'i (Lifting State Up)
  const [isSpeedMenuOpen, setIsSpeedMenuOpen] = useState(false); 

  // Ref'ler
  const mediaRecorderRef = useRef(null);
  const recordingChunksRef = useRef([]);
  const recordingIntervalRef = useRef(null);
  const streamRef = useRef(null);
  const audioElementRef = useRef(null);
  const playbackAnimationRef = useRef(null);
  
  // Görselleştirme Ref'leri
  const allRecordingDataRef = useRef([]);
  const updateCounterRef = useRef(0);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const timeDataRef = useRef(null);
  const animationFrameRef = useRef(null);

  const BAR_COUNT = 60; 

  // --- Logic Functions ---
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
        const rms = Math.sqrt(sum / blockSize);
        samples.push(rms);
      }

      const maxVal = Math.max(...samples, 0.01); 
      const normalizedSamples = samples.map(s => {
          const normalized = (s / maxVal); 
          const boosted = Math.pow(normalized, 0.8); 
          return Math.max(0.05, Math.min(1, boosted));
      });

      setSavedLevels(normalizedSamples);

    } catch (err) {
      console.error("Ses dosyası işlenirken hata:", err);
      alert("Ses dosyası analiz edilemedi.");
    }
  };

  // --- Visualizer & Recording ---
  const startVisualizer = () => {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const draw = () => {
      animationFrameRef.current = requestAnimationFrame(draw);
      const bufferLength = analyser.fftSize;
      if (!timeDataRef.current || timeDataRef.current.length !== bufferLength) {
        timeDataRef.current = new Uint8Array(bufferLength);
      }
      const timeData = timeDataRef.current;
      analyser.getByteTimeDomainData(timeData);

      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        const centered = (timeData[i] - 128) / 128;
        sum += centered * centered;
      }
      let rms = Math.sqrt(sum / bufferLength);
      rms = Math.pow(rms, 0.6) * 2.5; 

      allRecordingDataRef.current.push(rms);

      updateCounterRef.current++;
      if (updateCounterRef.current % 5 === 0) {
        setSavedLevels(prev => { 
             const currentView = allRecordingDataRef.current.slice(-BAR_COUNT);
             while(currentView.length < BAR_COUNT) currentView.unshift(0.02);
             return currentView;
        });
      }
    };
    draw();
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      recordingChunksRef.current = [];
      allRecordingDataRef.current = []; 
      
      setRecordingTime(0);
      setRecordedBlob(null);
      setRecordedUrl(null);
      setSavedLevels(new Array(BAR_COUNT).fill(0.02)); 

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordingChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(recordingChunksRef.current, { type: "audio/webm" });
        setRecordedBlob(blob);
        setRecordedUrl(URL.createObjectURL(blob));
        
        if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
        
        const fullWaveform = compressWaveform(allRecordingDataRef.current, BAR_COUNT);
        setSavedLevels(fullWaveform);
        
        if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
        if (audioContextRef.current) audioContextRef.current.close();
      };

      mediaRecorder.start();
      setIsRecording(true);

      recordingIntervalRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const audioContext = new AudioCtx();
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;
      
      startVisualizer();

    } catch (err) {
      console.error(err);
      alert("Mikrofona erişilemiyor.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(recordingIntervalRef.current);
    }
  };

  const resetRecording = () => {
    setRecordedBlob(null);
    setRecordedUrl(null);
    setRecordingTime(0);
    setSavedLevels([]);
    allRecordingDataRef.current = [];
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

  // useCallback ile tanımlandı, hata çözüldü
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
    setSavedLevels([]);
    setRecordingTime(0);
    setIsRecording(false);
    setPlayProgress(0);
    setIsPlaying(false);
    setIsSpeedMenuOpen(false);
  };

  const handleAnalyze = () => {
    alert("Ses analizi başlatılıyor... (Backend bağlantısı eklenecek)");
  };

  useEffect(() => {
    return () => {
      clearInterval(recordingIntervalRef.current);
      if (playbackAnimationRef.current) cancelAnimationFrame(playbackAnimationRef.current);
    };
  }, []);

  return (
    <div className="relative w-full flex flex-col items-center justify-center overflow-hidden bg-slate-900 font-sans selection:bg-indigo-500 selection:text-white">
      
      {/* Arka Plan */}
      <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-indigo-600 rounded-full mix-blend-multiply filter blur-[128px] opacity-40 animate-blob"></div>
      <div className="absolute top-[-10%] right-[-10%] w-96 h-96 bg-purple-600 rounded-full mix-blend-multiply filter blur-[128px] opacity-40 animate-blob animation-delay-2000"></div>
      <div className="absolute -bottom-32 left-20 w-96 h-96 bg-pink-600 rounded-full mix-blend-multiply filter blur-[128px] opacity-40 animate-blob animation-delay-4000"></div>

      <div className="relative z-10 w-full max-w-5xl px-6 flex flex-col items-center">
        
      {/* Başlık */}
        <div className="relative text-center mb-10 -top-5">
          <h1 className="py-6 pb-8 text-4xl md:text-6xl font-extrabold text-transparent bg-clip-text bg-linear-to-r from-white via-indigo-100 to-indigo-200 tracking-tight overflow-visible">
            Sesini Keşfet
          </h1>
        </div>

        {/* Kart - Boyutu küçültüldü: min-h-[300px] */}
        <div className={`w-full bg-slate-800/40 backdrop-blur-xl border border-white/10 rounded-3xl shadow-2xl transition-all duration-500 min-h-[400px] flex flex-col relative ${isSpeedMenuOpen ? 'overflow-visible' : 'overflow-hidden'}`}>
          
          {/* Geri Dön Butonu */}
          {mode && (
            <button 
              onClick={goBack} 
              className="absolute top-4 left-4 z-50 w-10 h-10 grid place-items-center rounded-full bg-slate-700/30 hover:bg-slate-600/50 text-slate-300 hover:text-white transition-all duration-300 border border-white/5 group"
            >
              <FaChevronLeft className="text-base group-hover:scale-110 transition-transform" />
            </button>
          )}

          <div className="flex-1 p-8 flex flex-col items-center justify-center h-full">
            
            {/* 1. Mod Seçimi */}
            {!mode && <ModeSelection onSelectMode={setMode} />}

            {/* 2. Dosya Yükleme */}
            {mode === "upload" && !recordedUrl && (
              <FileUploader 
                onFileSelect={processUploadedFile} 
                isDragging={isDragging}
                setIsDragging={setIsDragging}
              />
            )}

            {/* 3. Oynatıcı / Kayıt Arayüzü */}
            {((mode === "record") || (mode === "upload" && recordedUrl)) && (
              <AudioPlayer 
                mode={mode}
                isRecording={isRecording}
                recordedUrl={recordedUrl}
                recordingTime={recordingTime}
                levels={savedLevels}
                isPlaying={isPlaying}
                playProgress={playProgress}
                playbackRate={playbackRate}
                onStartRecording={startRecording}
                onStopRecording={stopRecording}
                onTogglePlay={togglePlayPause}
                onSpeedChange={handleSpeedChange}
                onAnalyze={handleAnalyze}
                isSpeedMenuOpen={isSpeedMenuOpen}
                setIsSpeedMenuOpen={setIsSpeedMenuOpen}
              />
            )}

            {/* Gizli Audio Elementi */}
            {recordedUrl && (
              <audio
                ref={audioElementRef}
                src={recordedUrl}
                onEnded={() => { 
                  setIsPlaying(false); 
                  setPlayProgress(0); 
                  if (playbackAnimationRef.current) cancelAnimationFrame(playbackAnimationRef.current);
                  if (audioElementRef.current) audioElementRef.current.currentTime = 0; 
                }}
                className="hidden"
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero;