import { useTranslation } from 'react-i18next';
import { useTheme } from '../context/ThemeContext';
import { FiHeadphones, FiActivity, FiZap } from 'react-icons/fi';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';


/* ── use case data ── */
const CASES = [
    {
        icon: <FiHeadphones size={32} />,
        iconColor: '#bd9dff',
        iconGlow: 'rgba(189,157,255,0.3)',
        titleKey: 'usecase_1_title',
        descKey: 'usecase_1_desc',
    },
    {
        icon: <FiActivity size={32} />,
        iconColor: '#34b5fa',
        iconGlow: 'rgba(52,181,250,0.3)',
        titleKey: 'usecase_2_title',
        descKey: 'usecase_2_desc',
    },
    {
        icon: <FiZap size={32} />,
        iconColor: '#ec63ff',
        iconGlow: 'rgba(236,99,255,0.3)',
        titleKey: 'usecase_3_title',
        descKey: 'usecase_3_desc',
    },
];

const fadeUp = {
    initial: { opacity: 0, y: 24 },
    whileInView: { opacity: 1, y: 0 },
    viewport: { once: true },
    transition: { duration: 0.8, ease: [0.16, 1, 0.3, 1] as any },
};

const UseCasesPage = () => {
    const { t } = useTranslation();
    const { isDark } = useTheme();
    const navigate = useNavigate();

    /* theme-aware tokens */
    const onSurface = isDark ? '#dee5ff' : '#1a1a2e';
    const onSurfaceVariant = isDark ? '#a3aac4' : '#64748b';

    return (
        <div
            className="w-full flex-grow flex flex-col items-center relative"
        >

            <div className="w-full max-w-6xl px-4 md:px-10 relative z-10 flex flex-col pb-24" style={{ paddingTop: '7rem' }}>

                {/* ══════════ HERO ══════════ */}
                <motion.div {...fadeUp} className="mb-16 mx-4 md:mx-0">
                    <div
                        className="relative rounded-xl overflow-hidden flex flex-col items-center justify-center min-h-[380px] p-8 text-center"
                        style={{
                            boxShadow: '0 20px 40px rgba(0,0,0,0.4)',
                            backgroundColor: isDark ? '#060e20' : '#f8faff',
                        }}
                    >
                        {/* Video Background */}
                        <video 
                            autoPlay 
                            loop 
                            muted 
                            playsInline 
                            className={`absolute inset-0 w-full h-full object-cover z-0 ${isDark ? 'opacity-50 mix-blend-screen' : 'opacity-20'}`}
                        >
                            <source src="/Soundwave-Loop.mp4" type="video/mp4" />
                        </video>

                        {/* Gradient Overlay for text readability */}
                        <div className="absolute inset-0 z-0"
                            style={{
                                backgroundImage: `linear-gradient(to bottom, ${isDark ? 'rgba(6,14,32,0.5)' : 'rgba(248,250,255,0.6)'}, ${isDark ? 'rgba(6,14,32,0.92)' : 'rgba(248,250,255,0.95)'})`
                            }} 
                        />
                        {/* inner top glow border */}
                        <div className="absolute inset-0 pointer-events-none"
                            style={{ boxShadow: 'inset 0 1px 0 rgba(64,72,93,0.15)' }} />

                        <div className="max-w-3xl flex flex-col gap-4 items-center z-10 relative">
                            <h1
                                className="text-4xl md:text-5xl lg:text-6xl font-black leading-tight tracking-tight"
                                style={{ color: onSurface }}
                            >
                                {t('usecase_hero_title')}
                            </h1>
                            <p className="text-base md:text-lg max-w-2xl mt-2 leading-relaxed"
                                style={{ color: onSurfaceVariant }}>
                                {t('usecase_hero_subtitle')}
                            </p>
                        </div>
                    </div>
                </motion.div>

                <div className="w-full h-16 md:h-24"></div>

                {/* ══════════ CORE SCENARIOS ══════════ */}
                <motion.div {...fadeUp}>
                    {/* section header */}
                    <div className="flex flex-col gap-3 mb-8 px-1">
                        <h2 className="text-3xl font-bold tracking-tight whitespace-nowrap"
                            style={{ color: onSurface }}>
                            {t('usecase_core_scenarios')}
                        </h2>
                        <p className="text-sm md:text-base leading-relaxed max-w-3xl"
                            style={{
                                color: onSurfaceVariant,
                                borderLeft: `3px solid rgba(189,157,255,0.5)`,
                                paddingLeft: '0.85rem',
                            }}>
                            {t('usecase_core_scenarios_note')}
                        </p>
                    </div>

                    {/* cards grid */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-2 mx-4 md:mx-0">
                        {CASES.map((item, idx) => (
                            <div
                                key={idx}
                                className="group relative flex flex-col gap-6 rounded-none transition-all duration-300 backdrop-blur-xl"
                                style={{
                                    padding: '1.5rem 2rem',
                                    background: isDark ? 'rgba(9, 19, 40, 0.6)' : 'rgba(255, 255, 255, 0.45)',
                                    boxShadow: 'inset 0 1px 0 rgba(64,72,93,0.15)',
                                    cursor: 'default',
                                }}
                                onMouseEnter={e => {
                                    (e.currentTarget as HTMLDivElement).style.background = isDark ? 'rgba(20, 31, 56, 0.8)' : 'rgba(255, 255, 255, 0.8)';
                                    (e.currentTarget as HTMLDivElement).style.transform = 'scale(1.02)';
                                }}
                                onMouseLeave={e => {
                                    (e.currentTarget as HTMLDivElement).style.background = isDark ? 'rgba(9, 19, 40, 0.6)' : 'rgba(255, 255, 255, 0.45)';
                                    (e.currentTarget as HTMLDivElement).style.transform = 'scale(1)';
                                }}
                            >
                                {/* icon */}
                                <div
                                    className="mb-2 transition-transform duration-300 group-hover:-translate-y-1"
                                    style={{
                                        marginTop: '0.5rem',
                                        color: item.iconColor,
                                        filter: `drop-shadow(0 0 12px ${item.iconGlow})`,
                                    }}
                                >
                                    {item.icon}
                                </div>

                                {/* text */}
                                <div className="flex flex-col gap-2">
                                    <h3 className="text-xl font-bold leading-tight"
                                        style={{ color: onSurface }}>
                                        {t(item.titleKey)}
                                    </h3>
                                    <p className="text-sm leading-relaxed"
                                        style={{ color: onSurfaceVariant }}>
                                        {t(item.descKey)}
                                    </p>
                                </div>
                            </div>
                        ))}
                    </div>
                </motion.div>

                <div className="w-full h-16 md:h-24"></div>

                {/* ══════════ CTA ══════════ */}
                <motion.div {...fadeUp} className="flex justify-center">
                    <div
                        className="flex flex-col items-center gap-8 text-center w-full max-w-4xl"
                    >
                        <div className="flex flex-col gap-3">
                            <h2 className="text-3xl md:text-4xl font-bold tracking-tight"
                                style={{ color: onSurface }}>
                                {t('ready_to_start')}
                            </h2>
                            <p className="text-base" style={{ color: onSurfaceVariant }}>
                                {t('cta_subtitle')}
                            </p>
                        </div>

                        <div className="flex flex-wrap gap-4 justify-center mt-6">
                            <button
                                type="button"
                                onClick={() => navigate('/technical-info')}
                                className="min-w-[180px] h-14 px-10 rounded-sm text-sm md:text-base font-bold tracking-wide transition-all hover:scale-105 active:scale-95 border border-purple-500/40 hover:border-purple-400/60 backdrop-blur-xl shadow-lg"
                                style={{
                                    background: isDark ? 'linear-gradient(135deg, rgba(147, 51, 234, 0.25) 0%, rgba(79, 70, 229, 0.25) 100%)' : 'linear-gradient(135deg, rgba(147, 51, 234, 0.15) 0%, rgba(79, 70, 229, 0.15) 100%)',
                                    color: '#ffffff',
                                    boxShadow: '0 8px 32px rgba(147, 51, 234, 0.2)',
                                }}
                            >
                                {t('view_technical_info')}
                            </button>
                        </div>
                    </div>
                </motion.div>

                {/* Additional spacer for footer clearance */}
                <div className="w-full h-24 md:h-32"></div>

            </div>
        </div>
    );
};

export default UseCasesPage;
