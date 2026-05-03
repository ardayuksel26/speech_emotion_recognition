import React, { useEffect, useRef } from 'react';
import { useTheme } from '../../context/ThemeContext';

interface WaveformVisualizerProps {
    audioUrl?: string;
    isRecording?: boolean;
    recordingStream?: MediaStream;
}

const WaveformVisualizer: React.FC<WaveformVisualizerProps> = ({
    isRecording,
    recordingStream
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { isDark } = useTheme();
    // Persistent data for scrolling effect
    const dataHistoryRef = useRef<number[]>([]);
    const animationRef = useRef<number | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const sourceRef = useRef<MediaStreamAudioSourceNode | MediaElementAudioSourceNode | null>(null);

    useEffect(() => {
        if (!canvasRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Initialize Audio Context
        if (!audioContextRef.current) {
            audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
        }

        // Canvas sizing
        const resizeCanvas = () => {
            const parentWidth = canvas.parentElement?.clientWidth || 600;
            canvas.width = parentWidth;
            canvas.height = 150; // Increased height for better detail

            // Fill history with zeros if empty or resize
            const neededPoints = Math.ceil(parentWidth / 2); // 1 point every 2 pixels
            if (dataHistoryRef.current.length < neededPoints) {
                const currentLen = dataHistoryRef.current.length;
                for (let i = 0; i < neededPoints - currentLen; i++) {
                    dataHistoryRef.current.unshift(0); // Fill transparently
                }
            }
        };
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        // Drawing function
        const draw = () => {
            if (!ctx) return;

            // Update Data (Push new sample to the END of history)
            if (analyserRef.current && isRecording) {
                const bufferLength = analyserRef.current.frequencyBinCount;
                const dataArray = new Uint8Array(bufferLength);
                analyserRef.current.getByteTimeDomainData(dataArray);

                // Calculate instant RMS volume
                let sum = 0;
                for (let i = 0; i < bufferLength; i++) {
                    const v = (dataArray[i] - 128) / 128.0;
                    sum += v * v;
                }
                const rms = Math.sqrt(sum / bufferLength);
                const value = Math.min(rms * 10, 1.0);
                dataHistoryRef.current.push(value);
            } else {
                dataHistoryRef.current.push(0);
            }

            // Keep history to half the canvas width worth of bars
            const sliceWidth = 3;
            const maxPoints = Math.ceil(canvas.width / 2 / sliceWidth) + 1;
            if (dataHistoryRef.current.length > maxPoints) {
                dataHistoryRef.current.shift();
            }

            // Draw Logic
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.lineWidth = 3;
            ctx.lineJoin = 'round';
            ctx.lineCap = 'round';

            // Gradient across full width
            const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
            gradient.addColorStop(0, '#6366f1');   // Indigo-500
            gradient.addColorStop(0.5, '#ec4899'); // Pink-500
            gradient.addColorStop(1, '#a855f7');   // Purple-500
            ctx.strokeStyle = gradient;

            const h = canvas.height;
            const centerY = h / 2;
            const centerX = canvas.width / 2;

            // The newest sample is at history[length-1] → drawn at centerX
            // Older samples go left
            ctx.beginPath();
            const history = dataHistoryRef.current;
            for (let i = 0; i < history.length; i++) {
                // i=0 is oldest (far left), i=length-1 is newest (center)
                const age = history.length - 1 - i; // 0 = newest, increases left
                const drawX = centerX - age * sliceWidth;
                if (drawX < 0) continue;

                const v = history[i];
                const barHeight = Math.max(v * h * 0.8, 2);

                ctx.moveTo(drawX, centerY - barHeight / 2);
                ctx.lineTo(drawX, centerY + barHeight / 2);
            }

            ctx.stroke();

            // Draw vertical playhead line at center
            ctx.save();
            ctx.beginPath();
            ctx.lineWidth = 2;
            ctx.strokeStyle = isDark ? 'rgba(255,255,255,0.5)' : 'rgba(99,102,241,0.7)';
            ctx.moveTo(centerX, 4);
            ctx.lineTo(centerX, h - 4);
            ctx.stroke();
            ctx.restore();

            animationRef.current = requestAnimationFrame(draw);
        };

        // Setup source based on props
        if (isRecording && recordingStream) {
            // Re-create analyser only if needed
            if (!analyserRef.current) {
                analyserRef.current = audioContextRef.current.createAnalyser();
                analyserRef.current.fftSize = 256;
            }

            // Re-connect source logic
            if (sourceRef.current) {
                sourceRef.current.disconnect();
            }
            sourceRef.current = audioContextRef.current.createMediaStreamSource(recordingStream);
            sourceRef.current.connect(analyserRef.current);

            draw();
        } else {
            // Continue drawing (scrolling silence) even if not recording? 
            // Or stop? The user asked for "flowing while audio recording". 
            // But if we stop, it freezes. Let's keep drawing to show "stopped" flatline or just stop properly.
            // If we stop start, we might loose the previous graph.
            // Let's keep loop running if we have context, but without pushing new data (or pushing 0).
            draw();
        }

        return () => {
            window.removeEventListener('resize', resizeCanvas);
            if (animationRef.current) cancelAnimationFrame(animationRef.current);
            // Don't close AudioContext generally, just disconnect nodes
            sourceRef.current?.disconnect();
        };
    }, [isRecording, recordingStream, isDark]);

    return (
        <div
            className="w-full h-32 rounded-2xl overflow-hidden shadow-inner relative"
            style={{
                background: isDark ? 'rgba(30,41,59,0.4)' : '#ffffff',
                border: isDark ? '1px solid rgba(255,255,255,0.1)' : '1px solid rgba(203,213,225,0.5)',
            }}
        >
            <canvas ref={canvasRef} className="w-full h-full drop-shadow-md" />
        </div>
    );
};

export default WaveformVisualizer;
