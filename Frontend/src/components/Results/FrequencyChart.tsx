import React from 'react';
import { useTranslation } from 'react-i18next';
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer
} from 'recharts';
import { useTheme } from '../../context/ThemeContext';

interface FrequencyChartProps {
    data: Array<{ time: number; freq: number }>;
}

export const FrequencyChart: React.FC<FrequencyChartProps> = ({ data }) => {
    const { t } = useTranslation();
    const { isDark } = useTheme();
    // Neutral color theme (Indigo)
    const color = { stroke: '#6366f1', fill: 'rgba(99, 102, 241, 0.2)' };

    return (
        <div className="w-full h-48 md:h-64 mt-4 animate-fadeIn">
            <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    <defs>
                        <linearGradient id="colorFreq" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={color.stroke} stopOpacity={0.3} />
                            <stop offset="95%" stopColor={color.stroke} stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid 
                        strokeDasharray="3 3" 
                        vertical={false} 
                        stroke={isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)"} 
                    />
                    <XAxis 
                        dataKey="time" 
                        hide 
                    />
                    <YAxis 
                        domain={[0, 'auto']} 
                        hide 
                    />
                    <Tooltip 
                        content={({ active, payload }) => {
                            if (active && payload && payload.length) {
                                return (
                                    <div className="bg-white dark:bg-slate-800 p-2 border border-slate-200 dark:border-slate-700 rounded-lg shadow-xl text-[10px] font-bold">
                                        <p className="text-slate-500 uppercase tracking-widest mb-1">{t('acoustic_pitch_contour')}</p>
                                        <p style={{ color: color.stroke }}>{payload[0].value.toFixed(1)} Hz</p>
                                    </div>
                                );
                            }
                            return null;
                        }}
                    />
                    <Area
                        type="monotone"
                        dataKey="freq"
                        stroke={color.stroke}
                        strokeWidth={3}
                        fillOpacity={1}
                        fill="url(#colorFreq)"
                        animationDuration={1500}
                    />
                </AreaChart>
            </ResponsiveContainer>
            <div className="flex justify-between mt-2 px-2">
                 <span className="text-[10px] font-bold uppercase tracking-widest opacity-40">{t('start')}</span>
                 <span className="text-[10px] font-black uppercase tracking-widest text-indigo-500/60 dark:text-indigo-400/60">{t('acoustic_pitch_contour')}</span>
                 <span className="text-[10px] font-bold uppercase tracking-widest opacity-40">{t('end')}</span>
            </div>
        </div>
    );
};
