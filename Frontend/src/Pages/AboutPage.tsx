import { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useTheme } from '../context/ThemeContext';
import { motion, AnimatePresence } from 'framer-motion';
import { FiAward, FiChevronLeft, FiChevronRight } from 'react-icons/fi';
import { FaGithub, FaLinkedin } from 'react-icons/fa';
import InteractiveBackground from '../components/InteractiveBackground';

/* ---------- types ---------- */
interface TeamMember {
    name: string;
    roleKey: string;
    descKey: string;
    photo: string;
    accent: string;
    accentText: string;
    github: string;
    linkedin: string;
    photoPosition?: string;
    photoSize?: string;
}

/* ---------- data ---------- */
const TEAM: TeamMember[] = [
    {
        name: 'Arda Yüksel',
        roleKey: 'about_member_1_role',
        descKey: 'about_member_1_desc',
        photo: '/team1.png',
        accent: 'from-violet-600 to-purple-500',
        accentText: 'text-violet-400',
        github: 'https://github.com/ardayuksel26',
        linkedin: 'https://www.linkedin.com/in/arda-yk26/',
    },
    {
        name: 'İlhan Uzunoğlu',
        roleKey: 'about_member_2_role',
        descKey: 'about_member_2_desc',
        photo: '/team2.png',
        accent: 'from-sky-500 to-cyan-400',
        accentText: 'text-sky-400',
        github: 'https://github.com/ilhanuzunoglu',
        linkedin: 'https://linkedin.com/in/ilhanuzunoglu',
        photoPosition: 'center top',
        photoSize: 'contain',
    },
    {
        name: 'Yağız Karhan Kökgül',
        roleKey: 'about_member_3_role',
        descKey: 'about_member_3_desc',
        photo: '/team3.png',
        accent: 'from-fuchsia-500 to-pink-400',
        accentText: 'text-fuchsia-400',
        github: 'https://github.com/karhankkgl',
        linkedin: 'https://www.linkedin.com/in/yagizkarhankokgul/',
    },
];

/* ---------- component ---------- */
const AboutPage = () => {
    const { t } = useTranslation();
    const { isDark } = useTheme();
    const [currentIndex, setCurrentIndex] = useState(0);
    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

    const startInterval = () => {
        if (intervalRef.current) clearInterval(intervalRef.current);
        intervalRef.current = setInterval(() => {
            setCurrentIndex((prev) => (prev + 1) % TEAM.length);
        }, 5000);
    };

    const handleNext = () => {
        setCurrentIndex((prev) => (prev + 1) % TEAM.length);
        startInterval();
    };

    const handlePrev = () => {
        setCurrentIndex((prev) => (prev - 1 + TEAM.length) % TEAM.length);
        startInterval();
    };

    useEffect(() => {
        startInterval();
        return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
    }, []);

    const fadeUp = {
        initial: { opacity: 0, y: 24 },
        whileInView: { opacity: 1, y: 0 },
        viewport: { once: true },
        transition: { duration: 0.8, ease: [0.16, 1, 0.3, 1] as any },
    };

    return (
        <div
            className="w-full flex-grow flex flex-col items-center relative"
            style={{ background: isDark ? '#060e20' : '#f8faff' }}
        >
            {/* Animated bubble background */}
            <InteractiveBackground />
            {/* ── ambient noise blobs ── */}
            <div className="fixed inset-0 pointer-events-none z-0">
                <div className="absolute top-[-5%] left-[-8%] w-[50%] h-[50%] rounded-full blur-[160px]"
                    style={{ background: 'rgba(189,157,255,0.08)' }} />
                <div className="absolute bottom-[5%] right-[-8%] w-[40%] h-[55%] rounded-full blur-[160px]"
                    style={{ background: 'rgba(52,181,250,0.07)' }} />
            </div>

            <div className="w-full max-w-6xl px-4 md:px-10 relative z-10 flex flex-col pb-32" style={{ paddingTop: '7rem' }}>

                {/* ════════════════════ HERO ════════════════════ */}
                <motion.div {...fadeUp} className="mb-24">
                    <div
                        className="relative rounded-2xl overflow-hidden flex flex-col items-center text-center"
                        style={{
                            boxShadow: '0 20px 40px rgba(0,0,0,0.4)',
                            paddingTop: '5rem',
                            paddingBottom: '5rem',
                            paddingLeft: '2.5rem',
                            paddingRight: '2.5rem',
                        }}
                    >
                        {/* Video Background */}
                        <video 
                            autoPlay 
                            loop 
                            muted 
                            playsInline 
                            className="absolute inset-0 w-full h-full object-cover z-0 opacity-80"
                            src="/Sounwave2.mp4"
                        />
                        {/* Dark/Light overlay so text remains readable */}
                        <div className={`absolute inset-0 z-0 ${isDark ? 'bg-[#060e20]/60' : 'bg-[#f8faff]/70'}`} />

                        {/* Glass panel inset border */}
                        <div className="absolute inset-0 rounded-2xl pointer-events-none z-0"
                            style={{ boxShadow: 'inset 0 1px 0 0 rgba(189,157,255,0.15)' }} />

                        {/* Hero content */}
                        <div className="max-w-3xl flex flex-col gap-6 items-center z-10 relative">
                            <h1
                                className="text-4xl md:text-6xl font-black leading-tight tracking-tight"
                                style={{
                                    background: 'linear-gradient(135deg, #bd9dff 0%, #34b5fa 100%)',
                                    WebkitBackgroundClip: 'text',
                                    WebkitTextFillColor: 'transparent',
                                    backgroundClip: 'text',
                                }}
                            >
                                {t('about_hero_tagline')}
                            </h1>
                            <p className={`text-lg md:text-xl leading-relaxed max-w-2xl ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                                {t('about_hero_subtitle')}
                            </p>
                        </div>
                    </div>
                </motion.div>

                <div className="w-full h-16 md:h-24"></div>

                {/* ════════════════════ INTRO TEXT ════════════════════ */}
                <motion.div {...fadeUp} className="mb-32 w-full flex flex-col items-center justify-center text-center relative z-10 px-4">
                    <p className={`max-w-4xl text-lg md:text-xl leading-relaxed ${isDark ? 'text-slate-300' : 'text-slate-700'} font-medium`}>
                        {t('about_custom_intro')}
                    </p>
                </motion.div>

                {/* ════════════════════ TEAM ════════════════════ */}
                <div className="w-full h-8 md:h-12"></div>
                <motion.div {...fadeUp} className="mb-10 w-full flex flex-col items-center justify-center">
                    <h2 className={`text-3xl font-black text-center mb-10 tracking-tight ${isDark ? 'text-white' : 'text-slate-900'} w-full`}>
                        {t('about_meet_team')}
                    </h2>

                    <div className="flex items-center justify-center w-full gap-4 md:gap-8 max-w-3xl mx-auto">
                        <button 
                            onClick={handlePrev} 
                            className={`w-12 h-12 flex items-center justify-center rounded-full flex-shrink-0 transition-all duration-200 border ${isDark ? 'bg-slate-800 border-slate-700 text-white hover:bg-slate-700 hover:scale-110' : 'bg-white border-slate-200 text-slate-800 hover:bg-slate-50 hover:scale-110 shadow-sm'}`}
                        >
                            <FiChevronLeft size={22} />
                        </button>

                        <div className="w-full max-w-sm relative overflow-hidden">
                            <div
                            className={`group relative flex flex-col rounded-xl overflow-hidden border ${
                                isDark
                                    ? 'border-white/10 shadow-2xl shadow-violet-500/10'
                                    : 'border-slate-200/40 shadow-xl shadow-violet-500/10'
                            }`}
                            style={{
                                minHeight: '460px',
                                background: isDark
                                    ? 'rgba(9, 19, 40, 0.55)'
                                    : 'rgba(255, 255, 255, 0.45)',
                                backdropFilter: 'blur(16px) saturate(180%)',
                                WebkitBackdropFilter: 'blur(16px) saturate(180%)',
                            }}>
                                {/* Top glow bar */}
                                <div className="absolute top-0 left-0 right-0 h-px opacity-20 z-10"
                                    style={{ background: 'linear-gradient(90deg, transparent, #bd9dff, transparent)' }} />

                                <AnimatePresence mode="wait">
                                    <motion.div
                                        key={currentIndex}
                                        initial={{ opacity: 0, x: 40 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        exit={{ opacity: 0, x: -40 }}
                                        transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
                                        className="absolute inset-0 flex flex-col"
                                    >
                                        {/* Photo */}
                                        <div className="relative overflow-hidden" style={{ height: '300px' }}>
                                            <div
                                                className="w-full h-full bg-no-repeat"
                                                style={{ 
                                                    backgroundImage: `url('${TEAM[currentIndex].photo}')`,
                                                    backgroundPosition: TEAM[currentIndex].photoPosition ?? 'center',
                                                    backgroundSize: TEAM[currentIndex].photoSize ?? 'cover',
                                                }}
                                            />
                                            <div className={`absolute inset-0 bg-gradient-to-t ${isDark ? 'from-[#091328] via-transparent' : 'from-white/20 via-transparent'} to-transparent`} />
                                        </div>

                                        {/* Info */}
                                        <div className="p-6 flex flex-col justify-center items-center text-center">
                                            <h3 className={`text-2xl font-bold leading-snug mb-1 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                                                {TEAM[currentIndex].name}
                                            </h3>
                                            <p className={`text-sm font-bold mb-3 ${TEAM[currentIndex].accentText}`}>
                                                {t(TEAM[currentIndex].roleKey)}
                                            </p>
                                            <p className={`text-sm leading-relaxed ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                                                {t(TEAM[currentIndex].descKey)}
                                            </p>
                                            {/* Social Links */}
                                            <div className="flex items-center gap-4 mt-5">
                                                <a
                                                    href={TEAM[currentIndex].github}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className={`p-2 rounded-full transition-all duration-200 hover:scale-110 ${isDark ? 'text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700' : 'text-slate-600 hover:text-slate-900 bg-slate-100 hover:bg-slate-200'}`}
                                                >
                                                    <FaGithub size={20} />
                                                </a>
                                                <a
                                                    href={TEAM[currentIndex].linkedin}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className={`p-2 rounded-full transition-all duration-200 hover:scale-110 ${isDark ? 'text-slate-400 hover:text-sky-400 bg-slate-800 hover:bg-slate-700' : 'text-slate-600 hover:text-sky-600 bg-slate-100 hover:bg-slate-200'}`}
                                                >
                                                    <FaLinkedin size={20} />
                                                </a>
                                            </div>
                                        </div>
                                    </motion.div>
                                </AnimatePresence>
                            </div>
                        </div>

                        <button 
                            onClick={handleNext} 
                            className={`w-12 h-12 flex items-center justify-center rounded-full flex-shrink-0 transition-all duration-200 border ${isDark ? 'bg-slate-800 border-slate-700 text-white hover:bg-slate-700 hover:scale-110' : 'bg-white border-slate-200 text-slate-800 hover:bg-slate-50 hover:scale-110 shadow-sm'}`}
                        >
                            <FiChevronRight size={22} />
                        </button>
                    </div>
                </motion.div>

                {/* ════════════════════ MISSION & VISION ════════════════════ */}
                <div className="w-full h-16 md:h-24"></div>
                <motion.div {...fadeUp} className="mb-24 px-0">
                    <div
                        className={`relative rounded-2xl overflow-hidden p-10 md:p-16 border ${
                            isDark ? 'border-white/5' : 'border-slate-200/60'
                        }`}
                        style={{
                            background: isDark
                                ? 'rgba(25,37,64,0.4)'
                                : 'rgba(248,250,255,0.8)',
                            backdropFilter: 'blur(12px) saturate(180%)',
                        }}
                    >
                        {/* Left accent bar removed per request */}

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-14">
                            {/* Mission */}
                            <div className="flex flex-col gap-5">

                                <div>
                                    <h2 className={`text-2xl font-bold mb-4 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                                        {t('about_mission_title')}
                                    </h2>
                                    <p className={`leading-relaxed ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                                        {t('about_mission_desc')}
                                    </p>
                                </div>
                            </div>

                            {/* Vision */}
                            <div className="flex flex-col gap-5">

                                <div>
                                    <h2 className={`text-2xl font-bold mb-4 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                                        {t('about_vision_title')}
                                    </h2>
                                    <p className={`leading-relaxed ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                                        {t('about_vision_desc')}
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </motion.div>

                <div className="w-full h-16 md:h-24"></div>

                {/* ════════════════════ UNIVERSITY BADGE ════════════════════ */}
                <motion.div
                    {...fadeUp}
                    className={`relative rounded-xl overflow-hidden p-10 flex flex-col items-center justify-center text-center gap-4 border ${
                        isDark
                            ? 'bg-violet-500/5 border-violet-500/20'
                            : 'bg-indigo-50 border-indigo-100'
                    }`}
                >
                    <div className={`p-4 rounded-full mb-2 ${isDark ? 'bg-violet-500/20 text-violet-400' : 'bg-white text-violet-500 shadow-sm'}`}>
                        <FiAward className="text-3xl" />
                    </div>
                    <div className="space-y-2">
                        <h3 className={`text-2xl font-bold tracking-tight px-4 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                            {t('about_university')}
                        </h3>
                        <p className={`text-sm ${isDark ? 'text-violet-300' : 'text-violet-600'}`}>
                            {t('about_class_of')}
                        </p>
                    </div>
                </motion.div>

            </div>
        </div>
    );
};

export default AboutPage;
