import React from 'react';
import { FaGithub, FaLinkedinIn, FaWhatsapp, FaEnvelope } from 'react-icons/fa';

export const Footer: React.FC = () => {
    const currentYear = new Date().getFullYear();

    return (
        <footer className="relative z-10 py-10 px-5 border-t border-glass-border bg-gradient-to-b from-transparent to-black/80 w-full mt-auto">
            <div className="max-w-[800px] mx-auto text-center">
                <p className="text-base text-text-muted mb-5">
                    Crafted by <strong className="font-semibold bg-gradient-to-br from-primary-light to-secondary bg-clip-text text-transparent">Md Emon Hasan</strong>
                </p>

                <div className="flex justify-center gap-4 mb-6">
                    <SocialLink href="https://github.com/Md-Emon-Hasan" icon={<FaGithub />} type="github" />
                    <SocialLink href="https://www.linkedin.com/in/md-emon-hasan" icon={<FaLinkedinIn />} type="linkedin" />
                    <SocialLink href="https://wa.me/8801834363533" icon={<FaWhatsapp />} type="whatsapp" />
                    <SocialLink href="mailto:emon.mlengineer@gmail.com" icon={<FaEnvelope />} type="email" />
                </div>

                <p className="text-sm text-white/40">
                    © {currentYear} Translatica. All rights reserved.
                </p>
            </div>
        </footer>
    );
};

interface SocialLinkProps {
    href: string;
    icon: React.ReactNode;
    type: 'github' | 'linkedin' | 'whatsapp' | 'email';
}

const SocialLink: React.FC<SocialLinkProps> = ({ href, icon, type }) => {
    const styles = {
        github: 'from-[#333] to-[#24292e] shadow-[0_8px_20px_rgba(36,41,46,0.4)] hover:shadow-[0_12px_30px_rgba(36,41,46,0.6)] before:from-[#444] before:to-[#333]',
        linkedin: 'from-[#0077b5] to-[#005885] shadow-[0_8px_20px_rgba(0,119,181,0.4)] hover:shadow-[0_12px_30px_rgba(0,119,181,0.6)] before:from-[#0088cc] before:to-[#0077b5]',
        whatsapp: 'from-[#25d366] to-[#128c7e] shadow-[0_8px_20px_rgba(37,211,102,0.4)] hover:shadow-[0_12px_30px_rgba(37,211,102,0.6)] before:from-[#2ee374] before:to-[#25d366]',
        email: 'from-[#ea4335] to-[#c5221f] shadow-[0_8px_20px_rgba(234,67,53,0.4)] hover:shadow-[0_12px_30px_rgba(234,67,53,0.6)] before:from-[#ff574d] before:to-[#ea4335]',
    };

    return (
        <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className={`w-[50px] h-[50px] rounded-2xl flex items-center justify-center text-xl text-white no-underline transition-all duration-400 relative overflow-hidden bg-gradient-to-br hover:-translate-y-1 hover:scale-110 group ${styles[type]}`}
        >
            <div className={`absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100 bg-gradient-to-br ${styles[type].split(' ').pop()}`} />
            <span className="relative z-10">{icon}</span>
        </a>
    );
};
