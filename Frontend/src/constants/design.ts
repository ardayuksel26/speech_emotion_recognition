export const EMOTION_COLORS = {
    angry: {
        primary: '#ef4444', // Red-500
        bg: '#fef2f2',      // Red-50
        text: '#991b1b',    // Red-800
        border: '#fca5a5',  // Red-300
        dark: {
            bg: 'rgba(239, 68, 68, 0.2)',
            text: '#fca5a5',
            border: 'rgba(239, 68, 68, 0.5)'
        }
    },
    calm: {
        primary: '#3b82f6', // Blue-500
        bg: '#eff6ff',      // Blue-50
        text: '#1e40af',    // Blue-800
        border: '#93c5fd',  // Blue-300
        dark: {
            bg: 'rgba(59, 130, 246, 0.2)',
            text: '#93c5fd',
            border: 'rgba(59, 130, 246, 0.5)'
        }
    },
    happy: {
        primary: '#eab308', // Yellow-500
        bg: '#fefce8',      // Yellow-50
        text: '#854d0e',    // Yellow-800
        border: '#fde047',  // Yellow-300
        dark: {
            bg: 'rgba(234, 179, 8, 0.2)',
            text: '#fde047',
            border: 'rgba(234, 179, 8, 0.5)'
        }
    },
    sad: {
        primary: '#a855f7', // Purple-500
        bg: '#faf5ff',      // Purple-50
        text: '#6b21a8',    // Purple-800
        border: '#d8b4fe',  // Purple-300
        dark: {
            bg: 'rgba(168, 85, 247, 0.2)',
            text: '#d8b4fe',
            border: 'rgba(168, 85, 247, 0.5)'
        }
    },
    neutral: { // Fallback
        primary: '#64748b',
        bg: '#f8fafc',
        text: '#334155',
        border: '#cbd5e1',
        dark: {
            bg: 'rgba(100, 116, 139, 0.2)',
            text: '#cbd5e1',
            border: 'rgba(100, 116, 139, 0.5)'
        }
    }
};

export type EmotionType = keyof typeof EMOTION_COLORS;
