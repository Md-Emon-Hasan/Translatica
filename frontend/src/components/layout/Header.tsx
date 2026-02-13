import React from 'react';
import { FaLanguage, FaMicrochip, FaBrain, FaRobot } from 'react-icons/fa';

export const Header: React.FC = () => {
    return (
        <div className="text-center mb-10 animate-fade-in-down w-full">
            <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-br from-primary to-secondary rounded-3xl flex items-center justify-center text-4xl text-white shadow-[0_20px_40px_rgba(99,102,241,0.3)] ring-1 ring-white/10 -rotate-6 transition-transform duration-300 hover:rotate-0 hover:scale-105">
                <FaLanguage />
            </div>

            <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold tracking-tight mb-2 bg-gradient-to-br from-white via-primary-light to-secondary bg-clip-text text-transparent">
                Translatica
            </h1>

            <p className="text-lg text-text-muted font-normal tracking-wider mb-5">
                AI-Powered Translation Engine
            </p>

            <div className="flex flex-wrap gap-3 justify-center">
                <Badge icon={<FaMicrochip />} label="LoRA" />
                <Badge icon={<FaBrain />} label="PEFT" />
                <Badge icon={<FaRobot />} label="Transformer" />
            </div>
        </div>
    );
};

interface BadgeProps {
    icon: React.ReactNode;
    label: string;
}

const Badge: React.FC<BadgeProps> = ({ icon, label }) => {
    return (
        <span className="bg-white/10 border border-white/20 px-4 py-2 rounded-full text-sm font-semibold text-white flex items-center gap-2 shadow-lg backdrop-blur-md transition-all duration-300 hover:bg-white/20 hover:border-primary hover:text-white hover:scale-105">
            <span className="text-primary-light text-base">{icon}</span>
            {label}
        </span>
    );
};
