import React, { useEffect, useRef } from 'react';
import { useTheme } from '../context/ThemeContext';

const InteractiveBackground: React.FC = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { isDark } = useTheme();

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let width = canvas.width = window.innerWidth;
        let height = canvas.height = window.innerHeight;

        const particles: { x: number, y: number, vx: number, vy: number, baseVx: number, baseVy: number, size: number, color: string }[] = [];
        const particleCount = 60;
        const colors = isDark
            ? ['#6366f1', '#a855f7', '#ec4899', '#3b82f6'] // darker mode glow
            : ['#818cf8', '#c084fc', '#f472b6', '#60a5fa']; // lighter mode glow

        for (let i = 0; i < particleCount; i++) {
            const baseVx = (Math.random() - 0.5) * 1.5;
            const baseVy = (Math.random() - 0.5) * 1.5;
            particles.push({
                x: Math.random() * width,
                y: Math.random() * height,
                vx: baseVx,
                vy: baseVy,
                baseVx,
                baseVy,
                size: Math.random() * 80 + 20,
                color: colors[Math.floor(Math.random() * colors.length)]
            });
        }

        let mouse = { x: width / 2, y: height / 2 };

        const handleMouseMove = (e: MouseEvent) => {
            mouse.x = e.clientX;
            mouse.y = e.clientY;
        };

        const handleResize = () => {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        };

        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('resize', handleResize);

        const animate = () => {
            ctx.clearRect(0, 0, width, height);

            // Draw a subtle gradient overlay tracing mouse
            const gradient = ctx.createRadialGradient(mouse.x, mouse.y, 0, mouse.x, mouse.y, 400);
            if (isDark) {
                gradient.addColorStop(0, 'rgba(99, 102, 241, 0.15)');
                gradient.addColorStop(1, 'rgba(15, 23, 42, 0)');
            } else {
                gradient.addColorStop(0, 'rgba(199, 210, 254, 0.4)');
                gradient.addColorStop(1, 'rgba(243, 244, 246, 0)');
            }
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, width, height);

            particles.forEach((p) => {
                // Repulsion: push particles away from mouse
                const dx = mouse.x - p.x;
                const dy = mouse.y - p.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 300 && dist > 0) {
                    const force = (1 - dist / 300) * 0.6;
                    p.vx -= (dx / dist) * force;
                    p.vy -= (dy / dist) * force;
                }

                // Gradually pull velocity back toward base so particles
                // resume natural drifting once mouse moves away
                p.vx += (p.baseVx - p.vx) * 0.03;
                p.vy += (p.baseVy - p.vy) * 0.03;

                // Clamp speed so particles don't fly off screen
                const speed = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
                if (speed > 5) {
                    p.vx = (p.vx / speed) * 5;
                    p.vy = (p.vy / speed) * 5;
                }

                p.x += p.vx;
                p.y += p.vy;

                // Bounce off edges
                if (p.x < 0 || p.x > width) { p.vx *= -1; p.baseVx *= -1; }
                if (p.y < 0 || p.y > height) { p.vy *= -1; p.baseVy *= -1; }

                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fillStyle = p.color;
                
                // Add blur specifically for these particles
                ctx.shadowBlur = 40;
                ctx.shadowColor = p.color;
                ctx.globalAlpha = isDark ? 0.4 : 0.6; // adjust transparency based on theme
                ctx.fill();
                ctx.globalAlpha = 1.0;
                ctx.shadowBlur = 0;
            });

            requestAnimationFrame(animate);
        };

        animate();

        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('resize', handleResize);
        };
    }, [isDark]);

    return (
        <canvas
            ref={canvasRef}
            className="fixed inset-0 w-full h-full pointer-events-none z-0"
            style={{ 
                filter: 'blur(3xl)',
                clipPath: 'inset(0)'
            }}
        />
    );
};

export default InteractiveBackground;
