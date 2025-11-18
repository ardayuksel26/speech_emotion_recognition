import { useState, useRef, useEffect } from "react";
import { FaMicrophone, FaUpload, FaRedo } from "react-icons/fa";

const Hero = () => {
  const [mode, setMode] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  // Kayıt için state'ler
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0); // saniye
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [recordedUrl, setRecordedUrl] = useState(null);
  const [savedLevels, setSavedLevels] = useState([]); // Kaydedilen waveform
  const [isPlaying, setIsPlaying] = useState(false);
  const [playProgress, setPlayProgress] = useState(0); // 0-1 arası

  const mediaRecorderRef = useRef(null);
  const recordingChunksRef = useRef([]);
  const recordingIntervalRef = useRef(null);
  const streamRef = useRef(null);
  const audioElementRef = useRef(null);

  // 🔊 Sağdan sola akan barlar için
  const BAR_COUNT = 100;
  const [levels, setLevels] = useState(() => new Array(BAR_COUNT).fill(0));
  const updateCounterRef = useRef(0);

  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const timeDataRef = useRef(null);
  const animationFrameRef = useRef(null);

  // ----------------- Upload tarafı -----------------
  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith("audio/")) {
      setSelectedFile(file);
    } else {
      alert("Lütfen bir ses dosyası seçin.");
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("audio/")) {
      setSelectedFile(file);
    } else {
      alert("Lütfen bir ses dosyası bırakın.");
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  // ----------------- Visualizer (ses → sağdan gelen barlar) -----------------
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

      // Zaman domeninde veri (dalga formu) al
      analyser.getByteTimeDomainData(timeData);

      // RMS (root mean square) ile tek bir “ses şiddeti” değeri hesapla
      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        const centered = (timeData[i] - 128) / 128; // -1..1
        sum += centered * centered;
      }
      let rms = Math.sqrt(sum / bufferLength); // 0..1

      // Biraz daha dramatik olsun
      rms = Math.pow(rms, 0.6) * 1.5;

      // Her 3 frame'de bir güncelle (yavaşlatma)
      updateCounterRef.current++;
      if (updateCounterRef.current % 3 === 0) {
        // Sağdan sola akış için: eski değerleri sola kaydır, yeni değeri en sağa koy
        setLevels((prev) => {
          const next = prev.slice(1);
          next.push(rms);
          return next;
        });
      }
    };

    draw();
  };

  const stopVisualizer = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    timeDataRef.current = null;

    // Barlar sönük dursun
    setLevels(new Array(BAR_COUNT).fill(0.05));
  };

  // ----------------- Kayıt başlat -----------------
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // MediaRecorder: gerçek kaydı tutuyor
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      recordingChunksRef.current = [];
      setRecordingTime(0);
      setRecordedBlob(null);
      setRecordedUrl(null);

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordingChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(recordingChunksRef.current, { type: "audio/webm" });
        setRecordedBlob(blob);
        const url = URL.createObjectURL(blob);
        setRecordedUrl(url);

        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
          streamRef.current = null;
        }

        stopVisualizer();
      };

      mediaRecorder.start();
      setIsRecording(true);

      // Süre sayacı
      recordingIntervalRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);

      // Web Audio: canlı ses analizi
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const audioContext = new AudioCtx();
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048; // daha smooth
      analyser.smoothingTimeConstant = 0.7;

      source.connect(analyser);
      analyserRef.current = analyser;

      startVisualizer();
    } catch (err) {
      console.error(err);
      alert("Mikrofona erişilemiyor. Lütfen tarayıcı izinlerini kontrol et.");
    }
  };

  // ----------------- Kayıt durdur -----------------
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      // Mevcut waveform'u kaydet
      setSavedLevels([...levels]);
      
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
        recordingIntervalRef.current = null;
      }
    }
  };

  // Tekrar kaydet
  const resetRecording = () => {
    setRecordedBlob(null);
    setRecordedUrl(null);
    setRecordingTime(0);
    setLevels(new Array(BAR_COUNT).fill(0));
    setSavedLevels([]);
    setPlayProgress(0);
    setIsPlaying(false);
  };

  // Audio element event handlers
  const handleAudioPlay = () => {
    setIsPlaying(true);
  };

  const handleAudioPause = () => {
    setIsPlaying(false);
  };

  const handleAudioTimeUpdate = (e) => {
    const audio = e.target;
    if (audio.duration) {
      setPlayProgress(audio.currentTime / audio.duration);
    }
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
    setPlayProgress(0);
  };

  // Cleanup
  useEffect(() => {
    return () => {
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
      }
      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state !== "inactive"
      ) {
        mediaRecorderRef.current.stop();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      stopVisualizer();
    };
  }, []);

  // 00:12 formatı
  const formatTime = (seconds) => {
    const m = String(Math.floor(seconds / 60)).padStart(2, "0");
    const s = String(seconds % 60).padStart(2, "0");
    return `${m}:${s}`;
  };

  return (
    <section className="w-full flex-1 h-full flex flex-col items-center justify-center px-6 bg-gray-100">
      {/* Başlık + Açıklama */}
      <div className="max-w-3xl text-center mb-12">
        <h2 className="text-4xl font-bold mb-6 text-gray-800">
          Duygunu Sesten Analiz Et
        </h2>
        <p className="text-lg text-gray-600 leading-relaxed">
          Ses kaydı alarak veya hazır bir ses dosyası yükleyerek duygusal
          analizi başlatabilirsin. Şimdilik sadece arayüz çalışıyor,
          arka planda model bağlantısı daha sonra eklenecek.
        </p>
      </div>

      {/* Mod Seçim Butonları */}
      <div className="flex flex-wrap gap-6 mb-12">
        <button
          onClick={() => setMode("upload")}
          className={`flex items-center gap-3 px-8 py-4 rounded-xl border-2 text-lg font-medium transition-all duration-200 ${
            mode === "upload"
              ? "bg-blue-600 text-white border-blue-600 shadow-lg"
              : "bg-white text-gray-800 border-gray-300 hover:border-blue-500 hover:shadow-md"
          }`}
        >
          <FaUpload className="text-xl" />
          Ses Dosyası Yükle
        </button>

        <button
          onClick={() => setMode("record")}
          className={`flex items-center gap-3 px-8 py-4 rounded-xl border-2 text-lg font-medium transition-all duration-200 ${
            mode === "record"
              ? "bg-blue-600 text-white border-blue-600 shadow-lg"
              : "bg-white text-gray-800 border-gray-300 hover:border-blue-500 hover:shadow-md"
          }`}
        >
          <FaMicrophone className="text-xl" />
          Ses Kaydet
        </button>
      </div>

      {/* İçerik Kartı */}
      <div className="w-full max-w-4xl bg-white shadow-lg rounded-2xl p-8">
        {/* Ses Yükleme Arayüzü */}
        {mode === "upload" && (
          <div className="flex flex-col gap-6">
            <div
              className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-200 ${
                isDragging
                  ? "border-blue-500 bg-blue-50"
                  : "border-gray-300 bg-gray-50 hover:border-blue-400 hover:bg-blue-25"
              }`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
            >
              <div className="mb-4">
                <FaUpload className="text-4xl text-gray-400 mx-auto mb-4" />
              </div>
              <p className="text-xl font-medium mb-2 text-gray-700">
                Ses dosyasını buraya sürükleyip bırak
              </p>
              <p className="text-gray-500 mb-6">
                veya klasörden bir dosya seç
              </p>
              <label className="inline-block px-6 py-3 rounded-lg bg-blue-600 text-white text-lg font-medium cursor-pointer hover:bg-blue-700 transition-colors">
                Dosya Seç
                <input
                  type="file"
                  accept="audio/*"
                  className="hidden"
                  onChange={handleFileSelect}
                />
              </label>
            </div>

            {selectedFile && (
              <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                <p className="text-green-800">
                  <span className="font-medium">Seçilen dosya:</span>{" "}
                  {selectedFile.name}
                </p>
              </div>
            )}

            <p className="text-sm text-gray-500 text-center">
              MP3, WAV, OGG ve diğer tüm standart ses formatları desteklenir.
            </p>
          </div>
        )}

        {/* Ses Kaydetme Arayüzü */}
        {mode === "record" && (
          <div className="flex flex-col items-center text-center gap-6 py-6">
            {/* Mikrofon + durum */}
            <div className="flex flex-col items-center gap-2">
              <FaMicrophone
                className={`text-6xl mx-auto mb-2 ${
                  isRecording ? "text-red-500 animate-pulse" : "text-blue-600"
                }`}
              />
              <p className="text-sm text-gray-600">
                {isRecording
                  ? "Kayıt devam ediyor..."
                  : recordedBlob
                  ? "Kayıt tamamlandı. İstersen tekrar kaydedebilirsin."
                  : "Kayda başlamak için aşağıdaki butona tıkla."}
              </p>
            </div>

            {/* 🔊 Sağdan sola akışlı waveform */}
            <div className="w-full max-w-xl">
              <div className="w-full h-32 rounded-xl bg-gray-200 flex items-end px-3 gap-0.5 overflow-hidden shadow-inner relative">
                {/* Kaydedilmiş waveform varsa onu göster, yoksa canlı levels */}
                {(savedLevels.length > 0 ? savedLevels : levels).map((level, index) => {
                  const totalBars = savedLevels.length > 0 ? savedLevels.length : levels.length;
                  // Her çubuğun üzerinden geçmesi için daha hassas hesaplama
                  const exactPlayPosition = playProgress * (totalBars - 1);
                  const currentPlayIndex = Math.round(exactPlayPosition);
                  const isCurrentBar = savedLevels.length > 0 && index === currentPlayIndex && isPlaying;
                  // Geçilen barlar için farklı renk
                  const isPassed = savedLevels.length > 0 && index < currentPlayIndex;
                  
                  return (
                    <div
                      key={index}
                      className={`flex-1 rounded-full transition-all ${
                        isCurrentBar 
                          ? 'bg-green-500 duration-75' 
                          : isPassed 
                          ? 'bg-blue-400 duration-150'
                          : 'bg-blue-500 duration-150'
                      }`}
                      style={{
                        height: `${10 + level * 110}px`, // min 10px, max ~120px
                        opacity: isCurrentBar ? 1 : isPassed ? 0.5 : 0.3 + level * 0.7,
                        transform: isCurrentBar ? 'scaleY(1.15)' : 'scaleY(1)',
                      }}
                    />
                  );
                })}
              </div>
              <p className="text-xs text-gray-500 mt-2">
                {savedLevels.length > 0 
                  ? "Kaydedilen ses dalgası. Oynatırken yeşil çubuk her barın üzerinden geçer."
                  : "Sağdaki çubuklar en yeni sesi temsil eder, dalga soldan akarak ilerler."}
              </p>
            </div>

            {/* Süre Göstergesi */}
            <div className="text-lg font-mono text-gray-800">
              Süre:{" "}
              <span className="font-bold">{formatTime(recordingTime)}</span>
            </div>

            {/* Kayıt / Durdur / Tekrar */}
            <div className="flex items-center gap-4">
              {!isRecording && (
                <button
                  onClick={startRecording}
                  className="px-6 py-3 rounded-lg bg-blue-600 text-white font-medium hover:bg-blue-700 transition-colors flex items-center gap-2"
                >
                  <FaMicrophone className="text-lg" />
                  Kaydı Başlat
                </button>
              )}

              {isRecording && (
                <button
                  onClick={stopRecording}
                  className="px-6 py-3 rounded-lg bg-red-600 text-white font-medium hover:bg-red-700 transition-colors"
                >
                  Kaydı Durdur
                </button>
              )}

              {!isRecording && recordedBlob && (
                <button
                  onClick={resetRecording}
                  className="px-4 py-3 rounded-lg bg-gray-200 text-gray-800 font-medium hover:bg-gray-300 transition-colors flex items-center gap-2"
                >
                  <FaRedo className="text-sm" />
                  Tekrar Kaydet
                </button>
              )}
            </div>

            {/* Kayıt sonucu dinleme */}
            {recordedUrl && (
              <div className="mt-4 w-full max-w-md mx-auto">
                <p className="text-gray-700 font-medium mb-2">
                  Kaydını dinleyebilirsin:
                </p>
                <audio 
                  ref={audioElementRef}
                  controls 
                  src={recordedUrl} 
                  className="w-full"
                  onPlay={handleAudioPlay}
                  onPause={handleAudioPause}
                  onTimeUpdate={handleAudioTimeUpdate}
                  onEnded={handleAudioEnded}
                />
              </div>
            )}

            <p className="text-xs text-gray-500">
              Not: Ses kaydı tarayıcı üzerinden alınmaktadır. Şimdilik sadece
              arayüz seviyesinde saklanıyor, sunucuya gönderilmiyor.
            </p>
          </div>
        )}
      </div>
    </section>
  );
};

export default Hero;
