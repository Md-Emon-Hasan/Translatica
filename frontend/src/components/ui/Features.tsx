import React from 'react';
import { FaBolt, FaShieldAlt, FaCheckCircle } from 'react-icons/fa';

export const Features: React.FC = () => {
    return (
        <div className="flex flex-wrap justify-center gap-5 mt-20 mb-20 animate-fade-in-up animation-delay-400">
            <FeatureItem icon={<FaBolt />} text="Lightning Fast" />
            <FeatureItem icon={<FaShieldAlt />} text="Secure & Private" />
            <FeatureItem icon={<FaCheckCircle />} text="High Accuracy" />
        </div>
    );
};

interface FeatureItemProps {
    icon: React.ReactNode;
    text: string;
}

const FeatureItem: React.FC<FeatureItemProps> = ({ icon, text }) => {
    return (
        <div className="bg-black/40 border border-white/10 px-5 py-3 rounded-2xl flex items-center justify-center gap-3 shadow-lg hover:bg-black/60 hover:border-primary/50 hover:scale-105 transition-all duration-300 group backdrop-blur-sm">
            <div className="text-secondary p-2 bg-secondary/10 rounded-full group-hover:text-primary group-hover:bg-primary/20 transition-colors">
                {icon}
            </div>
            <span className="text-white font-medium text-sm tracking-wide">{text}</span>
        </div>
    );
};
