import React, { useEffect, useRef } from 'react';
import { useTheme } from '../../context/ThemeContext';

interface WaveformVisualizerProps {
    audioUrl?: string;
    isRecording?: boolean;
    recordingStream?: MediaStream;
}

const WaveformVisualizer: React.FC<WaveformVisualizerProps> = ({
    audioUrl,
    isRecording,
    recordingStream
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { isDark } = useTheme();
    // Persistent data for scrolling effect
    const dataHistoryRef = useRef<number[]>([]);
    const animationRef = useRef<number>();
    const audioContextRef = useRef<AudioContext>();
    const analyserRef = useRef<AnalyserNode>();
    const sourceRef = useRef<MediaStreamAudioSourceNode | MediaElementAudioSourceNode>();

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

            // Update Data (Shift and Push)
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

                // Push normalized height (RMS * scale)
                // Use a non-linear scale for better visual sensitivity
                const value = Math.min(rms * 10, 1.0);

                dataHistoryRef.current.push(value);
            } else {
                // If not recording (paused) or no source, push silence
                dataHistoryRef.current.push(0);
            }

            // Keep history limits
            const maxPoints = Math.ceil(canvas.width / 3);
            if (dataHistoryRef.current.length > maxPoints) {
                dataHistoryRef.current.shift();
            }


            // Draw Logic
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.lineWidth = 3;
            ctx.lineJoin = 'round';
            ctx.lineCap = 'round';

            // Gradient
            const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
            gradient.addColorStop(0, '#6366f1'); // Indigo-500
            gradient.addColorStop(0.5, '#ec4899'); // Pink-500
            gradient.addColorStop(1, '#a855f7'); // Purple-500
            ctx.strokeStyle = gradient;
            ctx.fillStyle = gradient; // For fill if needed

            // Draw Line
            ctx.beginPath();
            const sliceWidth = 3;
            let x = canvas.width - (dataHistoryRef.current.length * sliceWidth);
            // Start from right side flowing left? Or standard scrolling right
            // Standard scrolling usually: New data appears on right, old slides left.
            // So we iterate history from left (old) to right (new).

            x = canvas.width - (dataHistoryRef.current.length * sliceWidth);
            if (x < 0) x = 0; // Should fill screen eventually

            const h = canvas.height;
            const centerY = h / 2;

            for (let i = 0; i < dataHistoryRef.current.length; i++) {
                const v = dataHistoryRef.current[i];
                const height = Math.max(v * h * 0.8, 2); // Min 2px height

                const drawX = x + (i * sliceWidth);

                // Draw a symmetric bar or line
                ctx.moveTo(drawX, centerY - height / 2);
                ctx.lineTo(drawX, centerY + height / 2);
            }

            ctx.stroke();

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
        <div className="w-full h-32 rounded-xl overflow-hidden bg-slate-50 dark:bg-white/5 backdrop-blur-sm relative">
            <canvas ref={canvasRef} className="w-full h-full" />
        </div>
    );
};

export default WaveformVisualizer;
